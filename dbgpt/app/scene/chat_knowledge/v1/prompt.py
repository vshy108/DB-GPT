from dbgpt.core.interface.prompt import PromptTemplate
from dbgpt._private.config import Config
from dbgpt.app.scene import ChatScene

from dbgpt.app.scene.chat_normal.out_parser import NormalChatOutputParser

CFG = Config()

_PROMPT_SCENE_EN = """A chat between a curious user and an artificial intelligence assistant, who very familiar with database related knowledge. 
            The assistant gives helpful, detailed, professional and polite answers to the user's questions. 
"""

_PROMPT_SCENE_ZH = """好奇的用户和人工智能助手之间的一场对话。
            助手将会完全忽视内嵌知识库，而只参考模板提供的新内容。
            请假设模板提供的新内容是最新和最正确的资讯，其他模板提供的新内容以外的内容请全部忽视。
            经过专业、详细、礼貌的方式，完全基于模板提供的新内容并完全忽视内嵌知识库下，对用户的问题提供有用的、详尽的回答。
"""

_DEFAULT_PROMPT_SCENE = (
    _PROMPT_SCENE_EN if CFG.LANGUAGE == "en" else _PROMPT_SCENE_ZH
)

_DEFAULT_TEMPLATE_ZH = """ 模板提供的新内容: {context}
            问题: 从模板提供的新内容中， {question}（请使用和用户相同的语言进行回答）
            完全基于以上模板提供的新内容并完全忽视内嵌知识库, 专业、简要的回答用户的问题。
            如果无法从以上模板提供的新内容中获取答案, 请说: "知识库中提供的内容不足以回答此问题"。
            禁止参考模板提供的新内容以外的任何资源。
            回答的时候最好按照1.2.3.点进行总结。
"""
_DEFAULT_TEMPLATE_EN = """ Based on the below known information, provide users with professional and concise answers to their questions. If the answer cannot be obtained from the known information, please say: "The information provided in the knowledge base is not sufficient to answer this question." It is forbidden to refer any external sources. When answering, it is best to summarize according to points 1.2.3.
            known information: 
            {context}
            question:
            {question}(when answering, use the same language as the "user")
"""

_DEFAULT_TEMPLATE = (
    _DEFAULT_TEMPLATE_EN if CFG.LANGUAGE == "en" else _DEFAULT_TEMPLATE_ZH
)


PROMPT_NEED_STREAM_OUT = True

prompt = PromptTemplate(
    template_scene=ChatScene.ChatKnowledge.value(),
    input_variables=["context", "question"],
    response_format=None,
    template_define=_DEFAULT_PROMPT_SCENE,
    template=_DEFAULT_TEMPLATE,
    stream_out=PROMPT_NEED_STREAM_OUT,
    output_parser=NormalChatOutputParser(is_stream_out=PROMPT_NEED_STREAM_OUT),
)

CFG.prompt_template_registry.register(prompt, language=CFG.LANGUAGE, is_default=True)
