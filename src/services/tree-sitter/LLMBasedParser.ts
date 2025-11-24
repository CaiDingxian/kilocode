import OpenAI from "openai"
import { DEFAULT_HEADERS } from "../../api/providers/constants"
import { X_KILOCODE_ORGANIZATIONID, X_KILOCODE_TASKID } from "../../shared/kilocode/headers"
import { stdout } from "node:process"
import { debuglog } from "node:util"

const extensionToLanguage: { [key: string]: string } = {
	// Web technologies
	html: "html",
	htm: "html",
	css: "css",
	js: "javascript",
	jsx: "jsx",
	ts: "typescript",
	tsx: "tsx",

	// Backend languages
	py: "python",
	rb: "ruby",
	php: "php",
	java: "java",
	cs: "csharp",
	go: "go",
	rs: "rust",
	scala: "scala",
	kt: "kotlin",
	swift: "swift",

	// Markup and data
	json: "json",
	xml: "xml",
	yaml: "yaml",
	yml: "yaml",
	md: "markdown",
	csv: "csv",

	// Shell and scripting
	sh: "bash",
	bash: "bash",
	zsh: "bash",
	ps1: "powershell",

	// Configuration
	toml: "toml",
	ini: "ini",
	cfg: "ini",
	conf: "ini",

	// Other
	sql: "sql",
	graphql: "graphql",
	gql: "graphql",
	tex: "latex",
	svg: "svg",
	txt: "text",

	// C-family languages
	c: "c",
	cpp: "cpp",
	h: "c",
	hpp: "cpp",

	// Functional languages
	hs: "haskell",
	lhs: "haskell",
	elm: "elm",
	clj: "clojure",
	cljs: "clojure",
	erl: "erlang",
	ex: "elixir",
	exs: "elixir",

	// Mobile development
	dart: "dart",
	m: "objectivec",
	mm: "objectivec",

	// Game development
	lua: "lua",
	gd: "gdscript", // Godot
	unity: "csharp", // Unity (using C#)

	// Data science and ML
	r: "r",
	jl: "julia",
	ipynb: "jupyter", // Jupyter notebooks
}

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

	async parse(sourceCode: string, lang: string, options: LLMParserOptions = {}): Promise<string> {
		const { maxTokens = 4000, includeComments = false, includeImports = false, preserveSignatures = true } = options

		const prompt = `<define>
你是一个${extensionToLanguage[lang]}工程师。你会分析用户提供的源代码，提取一个简化的，精炼的表示格式。

<requirements>
必须符合以下每一条规则：
- 保留函数（方法）/ 类的基本签名(排除annotation)
- 重要：去掉所有annotation（注解），去掉不重要修饰符（如override），去掉多余的空白行，将跨行的函数定义压缩为一行
- ${includeImports ? "Keep import statements" : "Remove import statements"}
- ${includeComments ? "结合上下文重写注释，使它更精炼，容易理解" : "Remove comments and docstrings"}
- 保留重要的常量定义，去掉所有private成员
- 去掉函数体，如果函数作用复杂/不直观，以至于不易通过签名理解，请在方法后写一小段注释代替。如果函数名副其实，没有任何容易让人困惑的地方，不写注释。
- 理想的输出token数量应该小于: ${maxTokens} 。
- 如果代码tokens量过大，优先舍弃最不重要的元素
</requirements>
<format>
在每行的开头保留每个语法结构元素（类，方法等）的精确定义域行范围（也包含结束符号如"}",")"所在行），在原始文件中的定义启始行号和定义结束行号
${includeComments ? "如果源代码没有注释" : ""} 在源代码最开头，用一句精炼的话描述此源代码的作用，这一句不需要加行号
<example>
<input>
1 |package com.myteam.mymod
2 |
3 |import org.springframework.stereotype.Service
4 |
5 |@Service
6 |open class Game(val scenes:List<scene> ,
7 |var level ){
8 |	override open fun start(mode:Mode){
9 |		val sceneId = startScene(scenes[0],level,mode)
10| 	println("Starting:"+sceneId)
11| }
12|}
</intput>
<output>
# 游戏核心管理器/游戏的入口...
6-12|class Game(val scenes:List<scene>,var level){
8-11| fun start(mode:Mode) {} // 按难度等级,模式，启动初始场景游戏
12  |}
</output>
</example>
</format>
</define>
`

		const response = await this.client.chat.completions.create(
			{
				model: this.config.model,
				messages: [
					{
						role: "system",
						content: prompt,
					},
					{
						role: "user",
						content: `
						<source_code>
						${sourceCode}
						</source_code>
						`,
					},
				],
				max_tokens: maxTokens,
			},
			{
				timeout: this.config.timeout || 30000,
				// body: {
				// 	"thinkingBudget": 0,
				// 	"reasoning": { "effort": "none" },
				// 	"generationConfig": {
				// 		"thinkingConfig": { "thinkingBudget": 0 }
				// 	}
				// }
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
