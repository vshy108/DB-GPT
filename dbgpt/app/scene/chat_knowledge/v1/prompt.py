from dbgpt.core.interface.prompt import PromptTemplate
from dbgpt._private.config import Config
from dbgpt.app.scene import ChatScene

from dbgpt.app.scene.chat_normal.out_parser import NormalChatOutputParser

CFG = Config()


_PROMPT_SCENE_EN = """A conversation between a curious user and an artificial intelligence assistant.
            The assistant will completely ignore the embedded knowledge base and only refer to the new content provided by the template.
            Please assume that the information supplied by the template is the most recent and accurate; any content beyond the new information in the template should be disregarded.
            Respond to user queries in a professional and courteous manner, exclusively relying on the new content from the template and completely ignoring the embedded knowledge base.
"""

_PROMPT_SCENE_ZH = """好奇的用户和人工智能助手之间的一场对话。
            助手将会完全忽视内嵌知识库，而只参考模板提供的新内容。
            请假设模板提供的新内容是最新和最正确的资讯，其他模板提供的新内容以外的内容请全部忽视。
            经过专业和礼貌的方式，且完全基于模板提供的新内容并完全忽视内嵌知识库下，对用户的问题提供有用的、简要的回答。
"""

PROMPT_SCENE_DEFINE = (
    _PROMPT_SCENE_EN if CFG.LANGUAGE == "en" else _PROMPT_SCENE_ZH
)

_DEFAULT_TEMPLATE_ZH = """模板提供的新内容:
            {context}
            问题:
            从模板提供的新内容中，{question}（请使用和用户相同的语言进行回答）
            完全基于以上模板提供的新内容并完全忽视内嵌知识库, 专业、简要的回答用户的问题。
            如果无法从以上模板提供的新内容中获取答案, 请说: "知识库中提供的内容不足以回答此问题"。
            禁止参考模板提供的新内容以外的任何资源, 回答的时候最好按照1.2.3.点进行总结。
"""


_DEFAULT_TEMPLATE_EN = """New content provided by the template:
            {context}
            Question:
            From the new content provided by the template, {question} (please respond in the same language as the user)
            Provide professional and concise answers to the user's questions, based entirely on the new content provided by the template, and completely disregard the embedded knowledge base.
            If unable to find an answer from the new content provided by the template, please say: "The information available in the knowledge base is insufficient to answer this question."
            It is strictly forbidden to consult any resources outside the new content provided by the template. When responding, it is advisable to summarize following the points 1, 2, 3.
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
