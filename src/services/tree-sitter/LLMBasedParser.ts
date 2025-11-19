import OpenAI from "openai"
import { DEFAULT_HEADERS } from "../../api/providers/constants"
import { X_KILOCODE_ORGANIZATIONID, X_KILOCODE_TASKID } from "../../shared/kilocode/headers"

export interface LLMParserConfig {
	apiKey: string
	baseUrl: string
	model: string
	kiloCodeOrganizationId?: string
	taskId?: string
	timeout?: number
}

export interface LLMParserOptions {
	maxTokens?: number
	includeComments?: boolean
	includeImports?: boolean
	preserveSignatures?: boolean
}

export class LLMBasedParser {
	private client: OpenAI
	private config: LLMParserConfig

	constructor(config: LLMParserConfig) {
		this.config = config
		this.client = new OpenAI({
			apiKey: config.apiKey,
			baseURL: config.baseUrl,
			defaultHeaders: {
				...DEFAULT_HEADERS,
				...(config.kiloCodeOrganizationId
					? { [X_KILOCODE_ORGANIZATIONID]: config.kiloCodeOrganizationId }
					: {}),
				...(config.taskId ? { [X_KILOCODE_TASKID]: config.taskId } : {}),
			},
		})
	}

	async parse(sourceCode: string, options: LLMParserOptions = {}): Promise<string> {
		const { maxTokens = 4000, includeComments = false, includeImports = true, preserveSignatures = true } = options

		const prompt = `<task>
你是一个严谨的代码分析助手。请分析提供的源代码，在保留其基本结构和逻辑的同时，提取一个简化的，整洁的表示。

<requirements>
1. 保留所有函数/类的签名 ${preserveSignatures ? "完整" : "仅当它包含重要的类型信息"}
2. ${includeImports ? "Keep import statements" : "Remove import statements"}
3. ${includeComments ? "结合上下文重写注释，使它更精炼，容易理解" : "Remove comments and docstrings"}
4. 保留重要的常量定义
5. 移除多余的空白行
6. 不保留方法体，若方法体中有重要逻辑，结合上下文重写为一小段注释
</requirements>

<options>
- 理想的输出token数量应该为: ${maxTokens} 以内。
- 如果代码tokens量过大，优先舍弃最不重要的元素
</options>

Return only the condensed code without any explanations or markdown formatting.
</task>

<format>
在每行代码的开头保留原始所在行的行号
${includeComments ? "如果源代码没有注释" : ""} 在源代码最开头，用一段精炼话描述此源代码的作用，这一句不需要加行号
<example>
# class Foo 类 
5 | class Foo {
6 | 	fun bar(){}
7 | }
</example>
</format>

<source_code>
${sourceCode}
</source_code>`

		const response = await this.client.chat.completions.create(
			{
				model: this.config.model,
				messages: [
					{
						role: "user",
						content: prompt,
					},
				],
				max_tokens: maxTokens,
				temperature: 0.1,
			},
			{
				timeout: this.config.timeout || 30000,
			},
		)

		const condensedCode = response.choices[0]?.message?.content
		if (!condensedCode) {
			throw new Error("LLM returned empty response")
		}

		return condensedCode.trim()
	}
}

// Factory function for convenience
export function createLLMBasedParser(config: LLMParserConfig): LLMBasedParser {
	return new LLMBasedParser(config)
}
