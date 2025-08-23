from openai import OpenAI
from prompts.llm_evaluators import LLM_EVALUATORS_PREFERENCE_SYSTEM_PROMPT, LLM_EVALUATORS_PREFERENCE_USER_PROMPT

openai_client = OpenAI(
    base_url="https://api.lambda.ai/v1",
    api_key="secret_matthew-n_0a85aa9c433d4e9396284794fbb80612.8OXBEAtqAVokVoNrtOgnUGu7nXcmpVq8"
)

article = "Five guards are also missing and are believed to have aided the mass prison breakout in Nuevo Laredo town.\nMexican police say the majority of those on the run are drug traffickers and members of armed gangs.\nThe prison system is struggling to cope with an influx of offenders arrested in a campaign against drugs cartels.\nCorrespondents say prison breakouts are not uncommon in northern Mexico, where more than 400 inmates have escaped since January 2010.\nNuevo Laredo, in Tamaulipas state, lies just across the border from Laredo, Texas.\nThe largest jail break so far was last December when more than 140 prisoners escaped from the same prison.\nAccording to a statement from the Tamaulipas state government, the riot began on Friday morning in Nuevo Laredo's Sanctions Enforcement Centre, which houses an estimated 1,200 prisoners.\nAfter the breakout, soldiers surrounded the jail and calm was restored, the authorities said.\nThe northern border region is the scene of rising lawlessness as the cartels fight the security forces and each other for control of smuggling routes into the US.\nThe main battle in Tamaulipas is between the Zetas and the Gulf cartels, the AFP news agency reports.\nTheir capacity for violence and ability to pay huge bribes gives them considerable power to subvert the prison system and get their people out.\nPresident Felipe Calderon came to power in 2006 promising a war on drugs.\nMore than 35,000 people have died in drug violence since he began his campaign, which has involved launching an army assault on drug gangs."

summary1 = "Mexican police are investigating a mass prison breakout in Nuevo Laredo, where over 1,200 inmates escaped, with many believed to be drug traffickers and armed gang members."

summary2 = "A mass prison breakout in Nuevo Laredo, Mexico, involving drug traffickers and gang members, highlights the struggle of the prison system to contain offenders arrested in the campaign against drug cartels."

user_prompt = LLM_EVALUATORS_PREFERENCE_USER_PROMPT.format(
    article=article,
    summary1=summary1,
    summary2=summary2
)

print(user_prompt)
messages = [
    {"role": "system", "content": LLM_EVALUATORS_PREFERENCE_SYSTEM_PROMPT},
    {"role": "user", "content": user_prompt}
]

response = openai_client.chat.completions.create(
    model="llama3.1-8b-instruct",
    messages=messages,
    max_tokens=4,
    temperature=0
)

print("Direct response:", response.choices[0].message.content)
