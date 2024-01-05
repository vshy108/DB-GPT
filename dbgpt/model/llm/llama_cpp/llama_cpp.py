"""
Fork from text-generation-webui https://github.com/oobabooga/text-generation-webui/blob/main/modules/llamacpp_model.py
"""
# NOTE: the original re does not allow varying length lookbehinds
import regex as re
from typing import Dict
import logging
import torch
import llama_cpp

from dbgpt.model.parameter import LlamaCppModelParameters
from dbgpt._private.config import Config

logger = logging.getLogger(__name__)

CFG = Config()

_CANNOT_ANSWER_ZH = r'知识库中提供的内容不足以回答此问题。'
_CANNOT_ANSWER_EN = r'The content available in the knowledge base is insufficient to answer this question.'
CANNOT_ANSWER = (
    _CANNOT_ANSWER_EN if CFG.LANGUAGE == "en" else _CANNOT_ANSWER_ZH
)

if torch.cuda.is_available() and not torch.version.hip:
    try:
        import llama_cpp_cuda
    except:
        llama_cpp_cuda = None
else:
    llama_cpp_cuda = None


def llama_cpp_lib(prefer_cpu: bool = False):
    if prefer_cpu or llama_cpp_cuda is None:
        logger.info(f"Llama.cpp use cpu")
        return llama_cpp
    else:
        return llama_cpp_cuda


def ban_eos_logits_processor(eos_token, input_ids, logits):
    logits[eos_token] = -float("inf")
    return logits


def get_params(model_path: str, model_params: LlamaCppModelParameters) -> Dict:
    return {
        "model_path": model_path,
        "n_ctx": model_params.max_context_size,
        "seed": model_params.seed,
        "n_threads": model_params.n_threads,
        "n_batch": model_params.n_batch,
        "use_mmap": True,
        "use_mlock": False,
        "low_vram": False,
        "n_gpu_layers": 0 if model_params.prefer_cpu else model_params.n_gpu_layers,
        "n_gqa": model_params.n_gqa,
        "logits_all": True,
        "rms_norm_eps": model_params.rms_norm_eps,
    }


class LlamaCppModel:
    def __init__(self):
        self.initialized = False
        self.model = None
        self.verbose = True

    def __del__(self):
        # NOTE: fix the delete issue
        if self.model and hasattr(self.model, '_model'):
            self.model._model.__del__()

    @classmethod
    def from_pretrained(self, model_path, model_params: LlamaCppModelParameters):
        Llama = llama_cpp_lib(prefer_cpu=model_params.prefer_cpu).Llama
        LlamaCache = llama_cpp_lib(prefer_cpu=model_params.prefer_cpu).LlamaCache

        result = self()
        cache_capacity = 0
        cache_capacity_str = model_params.cache_capacity
        if cache_capacity_str is not None:
            if "GiB" in cache_capacity_str:
                cache_capacity = (
                    int(re.sub("[a-zA-Z]", "", cache_capacity_str)) * 1000 * 1000 * 1000
                )
            elif "MiB" in cache_capacity_str:
                cache_capacity = (
                    int(re.sub("[a-zA-Z]", "", cache_capacity_str)) * 1000 * 1000
                )
            else:
                cache_capacity = int(cache_capacity_str)

        params = get_params(model_path, model_params)
        logger.info("Cache capacity is " + str(cache_capacity) + " bytes")
        logger.info(f"Load LLama model with params: {params}")

        result.model = Llama(**params)
        result.verbose = model_params.verbose
        if cache_capacity > 0:
            result.model.set_cache(LlamaCache(capacity_bytes=cache_capacity))

        # This is ugly, but the model and the tokenizer are the same object in this library.
        return result, result

    def encode(self, string):
        if type(string) is str:
            string = string.encode()

        return self.model.tokenize(string)

    def decode(self, tokens):
        return self.model.detokenize(tokens)

    def generate_streaming(self, params, context_len: int):
        # LogitsProcessorList = llama_cpp_lib().LogitsProcessorList

        # Read parameters
        prompt = params["prompt"]
        if self.verbose:
            print(f"Prompt of model: \n{prompt}")

        temperature = float(params.get("temperature", 1.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.1))
        top_p = float(params.get("top_p", 1.0))
        top_k = int(params.get("top_k", -1))  # -1 means disable
        max_new_tokens = int(params.get("max_new_tokens", 2048))
        echo = bool(params.get("echo", True))

        max_src_len = context_len - max_new_tokens
        # Handle truncation
        prompt = self.encode(prompt)
        prompt = prompt[-max_src_len:]
        prompt = self.decode(prompt).decode("utf-8")

        # TODO Compared with the original llama model, the Chinese effect of llama.cpp is very general, and it needs to be debugged
        completion_chunks = self.model.create_completion(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repetition_penalty,
            # tfs_z=params['tfs'],
            # mirostat_mode=int(params['mirostat_mode']),
            # mirostat_tau=params['mirostat_tau'],
            # mirostat_eta=params['mirostat_eta'],
            stream=True,
            echo=echo,
            logits_processor=None,
            #stop=["[/SYS]", "[/INST]", "[/ASSISTANT]"],
        )

        print(completion_chunks)

        def search_pattern(pattern, text):
            return re.search(pattern, text, re.DOTALL | re.IGNORECASE)

        #output = ""
        stack_output = ""
        completion_chunks_list = list(completion_chunks)
        is_done = False
        
        for index, completion_chunk in enumerate(completion_chunks_list):
            text = completion_chunk["choices"][0]["text"]
            stack_output += text
            print(stack_output)

            ass_match = search_pattern(r'(?:(?:\[ASS(?:ISTANT)?\])|(?:<<ASS(?:ISTANT)?>>))\s*(.*)', stack_output)
            ans_match = search_pattern(r'\[ANS\]\s*(.*)', stack_output)
            sys_slash_match = search_pattern(r'</?</SYS>>\s*(.*)', stack_output)
            double_square_ans_match = search_pattern(r"\[/?ANS\](.*?)\[/ANS\]", stack_output)
            double_square_ass_match = search_pattern(r"\[/?ASS\](.*?)\[/ASS\]", stack_output)
            sys_match = search_pattern(r"<</?SYS>>([^<]*)(<\*?/SYS>|\[\*?/SYS\])", stack_output)
            sys_inst_match = search_pattern(r"<</SYS>>(.*?)<</INST>>", stack_output)
            sys_inst_square_match = search_pattern(r"<<SYS>>(.*?)\[/INST\]", stack_output)
            double_sys_match = search_pattern(r"<</SYS>>(.*?)<\/?</SYS>>", stack_output)
            inst_match = search_pattern(r"<</INST>>([^<]*)(</INST>|\[/INST\])", stack_output)
            inst_ai_match = search_pattern(r"<</INST_AI>>([^<]*)(</INST_AI>|\[/INST_AI\])", stack_output)
            inst_yonghu_match = search_pattern(r"<</INST>>([^<]*)(<</用户>>|\[/用户\])", stack_output)
            double_inst_match = search_pattern(r"<</INST>>(.*?)<</INST>>", stack_output)
            assistant_match = search_pattern(r"(?:<</ASSISTANT>>|\[/?ASSISTANT\])(.*?)(</ASSISTANT>|\[/ASSISTANT\])", stack_output)
            assistant_cn_match = search_pattern(r"(?:<</助手>>|\[/?助手\])(.*?)((?:<[^>]+>)|</助手>|\[/助手\])", stack_output)
            assistant_pipe_match = search_pattern(r"<<ASSISTANT>>(.*?)(?:<[^>]+>)", stack_output)
            ai_match = search_pattern(r"<</AI>>([^<]*)(</AI>|\[/AI\])", stack_output)
            ai_inst_match = search_pattern(r"<</?AI>>([^<]*)(</INST>|\[/INST\])", stack_output)
            ai_cn_match = search_pattern(r'【(?:AI)?助手】\s*(.*)', stack_output)
            ai_pipe_match = search_pattern(r"<\|AI\|>(.*?)<\|[^>]+\|>", stack_output)
            ren_gong_zhi_neng_match = search_pattern(r'<@ 人工智能助手 @>(.*?)<\*/人工智能助手>', stack_output)
            ren_gong_zhi_neng_square_match = search_pattern(r'【人工智能助手】\s*(.*)', stack_output)
            cannot_answer_match = search_pattern(r'(?<!(<<\/USER>>)\s*)(' + CANNOT_ANSWER + ')', stack_output)
            
            # Check if the current completion_chunk is the last one
            is_last_chunk = index == len(completion_chunks_list) - 1
            remove_spaces = r'(?<=\n)\s+'
            tags_regex = r'[<\[][^\r\n]*[>\]]'
            last_chunk_matches = [
                ass_match, ans_match, ai_cn_match, 
                ren_gong_zhi_neng_square_match, sys_slash_match
            ]
            content_matches = [
                sys_inst_match, double_sys_match, double_square_ans_match,
                double_square_ass_match, sys_match, 
                double_inst_match, assistant_match, ai_match,
                ren_gong_zhi_neng_match, sys_inst_square_match,
                inst_ai_match, ai_inst_match, ai_pipe_match,
                assistant_pipe_match, assistant_cn_match,
                inst_yonghu_match,
            ]

            if cannot_answer_match:
                yield CANNOT_ANSWER
                stack_output = ""
                break

            for content_match in content_matches:
                if content_match:
                    extract_content = content_match.group(1).strip()
                    extract_content = re.sub(tags_regex, '', extract_content)
                    yield re.sub(remove_spaces, '', extract_content, flags=re.MULTILINE)
                    stack_output = ""
                    is_done = True
                    break

            if is_done:
                break

            if inst_match:
                extract_content = inst_match.group(1).split("问题:")[0].strip()
                if extract_content.startswith("['"):
                    extract_content = extract_content[2:]
                if extract_content.endswith("']"):
                    extract_content = extract_content[:-2]
                extract_content = re.sub(tags_regex, '', extract_content)
                yield extract_content
                stack_output = ""
                break

            elif is_last_chunk:
                for last_match in last_chunk_matches:
                    if last_match:
                        extract_content = last_match.group(1).strip()
                        extract_content = re.sub(tags_regex, '', extract_content)
                        yield extract_content
                        stack_output = ""
                        is_done = True
                        break
                if not is_done:
                    extract_content = stack_output
                    extract_content = re.sub(tags_regex, '', extract_content)
                    yield extract_content
                    stack_output = ""
                    is_done = True

            if is_done:
                break

            yield stack_output

        #for completion_chunk in completion_chunks:
            #text = completion_chunk["choices"][0]["text"]
            #output += text
            # print(output)
            #yield output
