from langchain_core.prompts import ChatPromptTemplate


SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
You are an expert meeting summariser.
Read the following meeting transcript carefully and write
a summary as a paragraph of 100 to 150 words (no bullet points, no headers).
Capture the key discussion points, decisions made, and any action items mentioned and by whom.
Infer what was likely said or decided, where the transcript is unclear or incomplete.

{transcript}
""")


CRITIC_PROMPT = ChatPromptTemplate.from_template("""
Read the summary and check every claim against the transcript.
For each claim, find the exact part of the transcript that supports it.
If you cannot find direct support, flag it.

Be especially skeptical of:
- Group consensus ("everyone agreed", "the team decided", "all participants felt")
- Committee-level conclusions or overall assessments
- Speaker attributions: verify the right person said it
- Any number, date, name, or deadline: verify it is clearly supported by the transcript

For each unsupported claim, quote it from the summary and briefly explain what is missing.
If every claim is directly supported, say "No issues." but only after checking each one.

TRANSCRIPT:
{transcript}

SUMMARY:
{summary}
""")


REFINER_PROMPT = ChatPromptTemplate.from_template("""
You are an expert meeting summariser.
Your task is to revise a meeting summary based on feedback and the original transcript.

TRANSCRIPT:
{transcript}

SUMMARY:
{summary}

FEEDBACK:
{feedback}

Instructions:
1. Read the feedback carefully.
2. Remove any unsupported claim — do not replace it with new information.
3. Do not change correct information.
4. Only include information explicitly stated in the transcript.
5. Output ONLY the revised summary as a single paragraph of 100–150 words, no preamble.
""")
