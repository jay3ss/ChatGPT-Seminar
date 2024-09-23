from pprint import pprint

pprint(
    {
        "input": "how can i leverage chain of thought in my llm-based applications to give the llm better reasoning? please give me acouple sample prompts",
        "context": [
            Document(
                metadata={},
                page_content="ability to process vast amounts of information and derive general principles. For example, Kepler\ncompressed thousands of measurements into Kepler’s three laws of planetary motion, which precisely\ndescribe the orbits of planets around the Sun (Russell, 1964). In critical decision-making, humans\nfind abstraction to be helpful since it provides a broader view of the environment. This work explores\nhow LLMs can tackle complex tasks involving many low-level details through a two-step process\nof abstraction-and-reasoning. The first step is to show LLMs how to step back through in-context\nlearning – prompting them to derive high-level abstractions such as concepts and principles for\na specific example. The second step is to leverage the reasoning ability to reason on top of the\nhigh-level concepts and principles. We use few-shot exemplar demonstrations to execute STEP-BACK\nPROMPTING on LLMs. We experiment across a range of tasks involving domain specific reasoning such as Physics and Chem-",
            ),
            Document(
                metadata={},
                page_content="slower than that in math domain: the most com-\npetitive approach only achieves 50% and35% on\nTheoremQA and SciBench, respectively, not to\nmention methods built on open-source LLMs. In-\nstead of developing an omniscient and proficientLLMs on reasoning tasks across various scientific\ndomains, we believe it is more practical to teach\nLLMs the ability to use domain-specific tools to\nfacilitate their reasoning abilities in some domain\nwhen external functions (toolset) are attached. A.2 Tool Learning\nLLMs, both proprietary ones and open-source ones,\ndemonstrate promising capabilities leveraging ex-\nternal tools to solve problems beyond their lim-\nits (Qin et al., 2023a). Combined with specific\ntools, these tool-augmented LLMs achieve great\nsuccess on various tasks such as machine learn-\ning (Wu et al., 2023; Shen et al., 2023; Patil et al.,\n2023; Yang et al., 2023; Liu et al., 2023), question\nanswering (Peng et al., 2023; Gou et al., 2023a),",
            ),
            Document(
                metadata={},
                page_content="in language models. 1.First, chain of thought, in principle, allows models to decompose multi-step problems into\nintermediate steps, which means that additional computation can be allocated to problems\nthat require more reasoning steps. 2.Second, a chain of thought provides an interpretable window into the behavior of the model,\nsuggesting how it might have arrived at a particular answer and providing opportunities\nto debug where the reasoning path went wrong (although fully characterizing a model’s\ncomputations that support an answer remains an open question). 3.Third, chain-of-thought reasoning can be used for tasks such as math word problems,\ncommonsense reasoning, and symbolic manipulation, and is potentially applicable (at least\nin principle) to any task that humans can solve via language. 4.Finally, chain-of-thought reasoning can be readily elicited in sufﬁciently large off-the-shelf\nlanguage models simply by including examples of chain of thought sequences into the",
            ),
            Document(
                metadata={},
                page_content="reasoning steps that lead to the ﬁnal answer for a problem. We will show that sufﬁciently large\n2\nlanguage models can generate chains of thought if demonstrations of chain-of-thought reasoning are\nprovided in the exemplars for few-shot prompting. Figure 1 shows an example of a model producing a chain of thought to solve a math word problem\nthat it would have otherwise gotten incorrect. The chain of thought in this case resembles a solution\nand can interpreted as one, but we still opt to call it a chain of thought to better capture the idea that it\nmimics a step-by-step thought process for arriving at the answer (and also, solutions/explanations\ntypically come after the ﬁnal answer (Narang et al., 2020; Wiegreffe et al., 2022; Lampinen et al.,\n2022, inter alia )). Chain-of-thought prompting has several attractive properties as an approach for facilitating reasoning\nin language models. 1.First, chain of thought, in principle, allows models to decompose multi-step problems into",
            ),
        ],
        "answer": AIMessage(
            content='Leveraging Chain-of-Thought (CoT) in your LLM-based applications is a great way to improve its reasoning capabilities. Here\'s how you can do it:\n\n**What is Chain-of-Thought?**\n\nChain-of-Thought is a technique where the model generates a step-by-step thought process or reasoning sequence that leads to the final answer. This allows the model to break down complex problems into manageable sub-problems, making it easier to reason and arrive at the correct solution.\n\n**How to Leverage Chain-of-Thought in your LLM-based applications:**\n\n1. **Provide demonstrations**: Include examples of CoT sequences in your few-shot prompting data. These can be generated by humans or other models.\n2. **Use specific prompts**: Design prompts that encourage the model to generate a CoT sequence, such as "Explain how you arrived at this answer" or "Break down the problem into smaller steps".\n3. **Train on CoT datasets**: Use datasets that include CoT sequences and train your LLM on them.\n\n**Sample Prompts:**\n\nHere are some sample prompts you can use to encourage Chain-of-Thought in your LLM:\n\n1. **Math Word Problems**:\n\t* "Solve the following math problem: [problem]. Explain how you arrived at this answer."\n\t* "Break down the solution into smaller steps: [problem]"\n2. **Reasoning and Logic**:\n\t* "Explain why [statement] is true/false."\n\t* "Justify your answer by breaking it down into smaller steps: [question]"\n3. **Scientific Reasoning**:\n\t* "Describe the process you would use to solve this problem: [problem]."\n\t* "Break down the solution into smaller steps, explaining each step in detail: [problem]"\n\nHere\'s an example of a prompt that encourages CoT for a math word problem:\n\n"Tom has 15 boxes of pens. Each box contains 25 pens. How many pens does Tom have in total? Explain how you arrived at this answer by breaking down the solution into smaller steps."\n\nThis prompt encourages the model to generate a CoT sequence, which would look something like this:\n\n1. "First, I need to find out how many boxes of pens Tom has."\n2. "Since each box contains 25 pens, I can multiply the number of boxes (15) by the number of pens per box (25)."\n3. "This gives me a total of 375 pens."\n\nBy providing these prompts and training your LLM on CoT datasets, you can improve its reasoning capabilities and make it more effective in solving complex problems.\n\n**Additional Tips:**\n\n* Use specific language to encourage CoT, such as "Explain how you arrived at this answer" or "Break down the solution into smaller steps".\n* Provide demonstrations of CoT sequences to help train your LLM.\n* Experiment with different prompts and datasets to find what works best for your application.\n\nI hope this helps! Let me know if you have any further questions.',
            additional_kwargs={},
            response_metadata={
                "model": "llama3.1:8b",
                "created_at": "2024-09-23T19:52:26.053036722Z",
                "message": {"role": "assistant", "content": ""},
                "done_reason": "stop",
                "done": True,
                "total_duration": 7662148210,
                "load_duration": 13028823,
                "prompt_eval_count": 975,
                "prompt_eval_duration": 220111000,
                "eval_count": 618,
                "eval_duration": 7303000000,
            },
            id="run-f83367e0-5e40-426a-a8c4-b5267d8c67d9-0",
            usage_metadata={
                "input_tokens": 975,
                "output_tokens": 618,
                "total_tokens": 1593,
            },
        ),
    }
)
