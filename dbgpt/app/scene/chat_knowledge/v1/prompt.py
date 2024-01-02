from dbgpt.core.interface.prompt import PromptTemplate
from dbgpt._private.config import Config
from dbgpt.app.scene import ChatScene

from dbgpt.app.scene.chat_normal.out_parser import NormalChatOutputParser

CFG = Config()

_PROMPT_SCENE_EN = """A dialogue between a curious user and an artificial intelligence assistant. 
            The assistant will completely ignore the embedded knowledge base and only reference the new content provided by the template. 
            Please assume that the new content provided by the template is the latest and most accurate information, and disregard any content outside of what the template provides. 
            In a professional, detailed, and polite manner, based entirely on the new content provided by the template and ignoring the embedded knowledge base, 
            provide the user with useful and comprehensive answers to their questions.
"""

_PROMPT_SCENE_ZH = """好奇的用户和人工智能助手之间的一场对话。
            助手将会完全忽视内嵌知识库，而只参考模板提供的新内容来对话。
            经过专业、礼貌的方式，完全基于模板提供的新内容并完全忽视内嵌知识库的情况下，对用户的问题提供有用的、简单的回答。
            人工智能助手严禁自问自答。
"""

PROMPT_SCENE_DEFINE = (
    _PROMPT_SCENE_EN if CFG.LANGUAGE == "en" else _PROMPT_SCENE_ZH
)

_DEFAULT_TEMPLATE_ZH = """模板提供的新内容:
            {context}
            问题:
            从模板提供的新内容中，{question}（请使用和用户相同的语言进行回答）
            完全基于以上模板提供的新内容和忽视内嵌知识库的方式去回答用户的问题。
            如果无法从以上模板提供的新内容中获取答案，请说: "知识库中提供的内容不足以回答此问题"。
            否则，当问题不明确时，请说: "请给予更明确的问句"，而不是产生问题去问用户。
            禁止参考模板提供的新内容以外的任何资源，回答的时候最好按照1.2.3.点进行总结。
"""

_DEFAULT_TEMPLATE_EN = """New content from the template:
            {context}
            question: 
            Based on the new content provided by the template, {question} (Please answer using the same language as the user)
            Based entirely on the new content provided by the template and completely disregarding the embedded knowledge base, provide a professional and concise answer to the user's question. 
            If unable to obtain an answer from the new content provided by the template, please say, 'The knowledge base content is not sufficient to answer this question.' 
            Prohibited from referencing any resources outside of the new content provided by the template, it is preferable to summarise the answer following the points 1.2.3.
"""

_DEFAULT_TEMPLATE = (
    _DEFAULT_TEMPLATE_EN if CFG.LANGUAGE == "en" else _DEFAULT_TEMPLATE_ZH
)


PROMPT_NEED_STREAM_OUT = True

prompt = PromptTemplate(
    template_scene=ChatScene.ChatKnowledge.value(),
    input_variables=["context", "question"],
    response_format=None,
    template_define=PROMPT_SCENE_DEFINE,
    template=_DEFAULT_TEMPLATE,
    stream_out=PROMPT_NEED_STREAM_OUT,
    output_parser=NormalChatOutputParser(is_stream_out=PROMPT_NEED_STREAM_OUT),
)

CFG.prompt_template_registry.register(prompt, language=CFG.LANGUAGE, is_default=True)
