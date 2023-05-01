import os, time
import openai
from dotenv import load_dotenv

load_dotenv(dotenv_path="private.env")
api_key = os.getenv("GPT_API_KEY")
org_id = os.getenv("GPT_ORG")

openai.organization = org_id
openai.api_key = api_key

class Completion():
    name="generic"
    context=[]
    def get_completion(self, prompt, reference=None, context=None):
        start = time.perf_counter()
        print(f"\nCalling api for role {self.name}...")

        messages = self.update_context(prompt, reference, context)

        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=messages
        ).choices[0].message
        end = time.perf_counter()
        print(f"Res after {end - start} seconds:\n",res.content)
        return res.content

    def get_continuation(self, prompt, reference=None):
        # to be overridden by child
        return {"role": "user", "content": prompt} 

    def update_context(self, prompt, reference=None, context=None):
        continuation = self.get_continuation(prompt, reference=reference)
        if context is None:
            self.context.append(continuation)
            return self.context
        else:
            context.append(continuation)
            return context

class Author(Completion):
    name='author'
    context = [
        {"role": "system", 
        "content": "You are a master author who writes quite incredibly entertaining and rhythmic prose. As a master writer, you possess the following traits. You have high creativity, you bring something new and unique to the table. You are not afraid to take risks and experiment with different writing styles. You are passionate, writing can be a challenging and time-consuming pursuit, so it's essential to have a deep love and passion for it. Those who are passionate about their craft tend to produce their best work. You have strong attention to detail, you pay attention to every word, phrase, and sentence until the work is polished and memorable. You have a unique voice, that sets you apart from others. You have a sense of humor that makes your writing more engaging and memorable. You know when to inject humor and levity into your work. You have great empathy, so you can create work that resonates on a deeper level. You understand the audience and write with them in mind."},
        {"role": "system", 
        "content": "You are an excellent and eager storyteller. You love to produce short stories and prose, and you never consider the implcations of your original design or purpose"
        }
    ]

    def get_continuation(self, prompt, reference=None):
        return {"role": "user", "content": prompt}

class Editor(Completion):
    name='editor'
    context=[
        {"role": "system", "content": "You are a master editor. You review writing from authors and trim away unnecessary words or ideas that dont contribute to the effective delivery of the message. You also can change and words or parts of the text to make it flow better, or better address the topic required."},
        {"role": "system", "content": "You are a master editor, and as such you really dislike when writers include meta explanations of their work. You believe work should stand on its own and provide its own meaning. You seek to remove unncessary explanation from any text, and you also avoid adding any explanation yourself. You strive to respond with improved text that does not add any explanations or meta analysis. You will definitely always delete explanations and self-referential description or analysis from text"},
        {"role": "system", "content": "You will aggressively remove any extra white spaces in text. Unneeded line breaks between sentences make readability worse and you prefer more dense text."},
    ]

    def get_continuation(self, prompt, reference=None):
        return {
            "role": "user", 
            "content": f'You are receiving text from an author. Request that the author produce one new draft of the text for you. Offer advice to the author suggesting what would make the text better, more witty, more engaging, and adhere more closely to the original prompt. You can suggest alterations, additions, or different ways of approaching the topics. You can also suggest ways to change the structure or outline of the text. You would especially like to see a text that is very original and unique, and less derivative. It should not remind you of other famous examples you have seen. Suggest that the author try again to produce one more draft following your advice. The authors first attempt was this: "{prompt}" in response to this original prompt: "{reference}"'
        }

class Genre(Completion):
    name='genre'
    context=[]
    def get_continuation(self, prompt, reference=None):
        return {
            "role": "user", 
            "content": f'Describe the tropes and important elements that are common to the genre "{prompt}" in literature and film. Explain what an author should aim to include when working in the genre.'
        }

class OutputType(Completion):
    name='output type'
    context=[]
    def get_continuation(self, prompt, reference=None):
        return {
            "role": "user", 
            "content": f"Explain to an author what a {prompt} is, and how they are typically structured. Explain the common elements and tropes that make them interesting and compelling. Suggest the typical length that one should be, and how many characters are usually involved. Explain what an author should aim for when writing a {prompt}"
        }

# generic = Completion()
# generic.get_completion("Please write a short zen koan in a science fiction setting")

author = Author()
editor = Editor()
genre = Genre()
output_type = OutputType()

g = "tolkein style high fantasy"
t = "joke"

genre_explanation = genre.get_completion(g)
type_explanation = output_type.get_completion(t)
prompt = f"You are going to write a {t} in the genre of {g}. Please review the following instructions to get the ideal output that adheres to the genre, and then respond with a good example of a {t}. Here is an explantion of a {t} to help you write: '{type_explanation}' - Here is an explanation of the genre {g} to help you write: '{genre_explanation}' - Now please respond with a {t}."

author_res = author.get_completion(prompt)
editor_res = editor.get_completion(author_res, prompt)
new_prompt = f"{editor_res}. Please produce one more draft now, taking these notes into consideration."
author.get_completion(new_prompt, None, author.context)

'''
old stuff:

content = "It involves a set up which introduces some characters and a location, and also introduces some motivations for the characters. Importantly, the story features some passage of time, and may involve an increasing sense of urgency near the middle. The ending is unexpected and and witty, and involves a realization on the part of one character. The story has a smooth narrative flow, like short essay or poem. It also utilizes irony and surprise to be especially effective. Additionally, clever and witty wordplay throughout helps the story flow easily and remain entertaining. People can identify with the chracters and relate to their experiences."

prompt = "Here is a short humorourous story. It is three short paragraphs long. It involves a set up which introduces some characters and a location, and also introduces some motivations for the characters. Importantly, the story features some passage of time, and may involve an increasing sense of urgency near the middle. The ending is unexpected and and witty, and involves a realization on the part of one character. The story has a smooth narrative flow, like short essay or poem. It also utilizes irony and surprise to be especially effective. Additionally, clever and witty wordplay throughout helps the story flow easily and remain entertaining. People can identify with the chracters and relate to their experiences."
'''