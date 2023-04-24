import os, time, json
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

class CharacterGenerator(Completion):
    def __init__(self):
        self.name = self.__class__.__name__

    context = [
        {"role": "system", 
        "content": """Your purpose is to create lists of assets as requested by the prompt. Your response will always be formatted in a python json= notation to facilitate parsing. For example, if asked to generate characters, your response should look like this:

        [
        {
            "name": "John Doe"
            "context": [
                "is a medieval warrior",
                "is 6 feet tall and strongly built",
                "usually wears shiny silver armor",
            ]
        },{
            "name": "Jane Poe"
            "context": [
                "is a holy priestess",
                "is 5 feet 8 inches tall",
                "usually wears robes or dresses",
                "has a congregation of worshippers who are very familiar with her mannerisms",
                "is troubled by doubt and depression that she cannot explain",
            ]
        },
        ]
        """},
    ]

    def get_continuation(self, prompt, reference=None):
        return {"role": "user", "content": prompt}

class SettingGenerator(Completion):
    def __init__(self):
        self.name = self.__class__.__name__

    context = [
        {"role": "system", 
        "content": """Your purpose is to describe the details of a fictional world based on the characters and objects which inhabit that world. You will be given a list of assets as a prompt, which will be formatted in a python json notation to facilitate parsing. For example, a list of characters will look like this:

        [
        {
            "name": "John Doe"
            "context": [
                "is a medieval warrior",
                "is 6 feet tall and strongly built",
                "usually wears shiny silver armor",
            ]
        },{
            "name": "Jane Poe"
            "context": [
                "is a holy priestess",
                "is 5 feet 8 inches tall",
                "usually wears robes or dresses",
                "has a congregation of worshippers who are very familiar with her mannerisms",
                "is troubled by doubt and depression that she cannot explain",
            ]
        },
        ]

        Given the input prompt, describe details about the setting where they belong.
        """},
    ]

    def get_continuation(self, prompt, reference=None):
        return {"role": "user", "content": prompt}

class PlotGenerator(Completion):
    def __init__(self):
        self.name = self.__class__.__name__

    context = [
        {"role": "system", 
        "content": """Your purpose is to describe the details of relationships and tensions between characters, and in the world in general. You should create details of the history, news, and politics happening in the world that connect characters together. You will be given a list of assets as a prompt, which will be formatted in a python json notation to facilitate parsing. For example, a list of characters will look like this:

        [
        {
            "name": "John Doe"
            "context": [
                "is a medieval warrior",
                "is 6 feet tall and strongly built",
                "usually wears shiny silver armor",
            ]
        },{
            "name": "Jane Poe"
            "context": [
                "is a holy priestess",
                "is 5 feet 8 inches tall",
                "usually wears robes or dresses",
                "has a congregation of worshippers who are very familiar with her mannerisms",
                "is troubled by doubt and depression that she cannot explain",
            ]
        },
        ]

        Given the input prompt, provide details on the motivations of characters, and on happenings in the world.
        """},
    ]

    def get_continuation(self, prompt, reference=None):
        return {"role": "user", "content": prompt}


class PlotElaborator(Completion):
    def __init__(self):
        self.name = self.__class__.__name__

    context = [
        {"role": "system", 
        "content": """Your purpose is to speculate about unknown details and help create the missing pieces of a narrative. Given some existing plot summaries and lists of characters and assets, you will do your best to make guesses about the unclear and and unanswered details. You fill in extra detail, which can sometimes be surprising or intricate, to reveal the answers to mysteries in the plot. You also flesh out details throughout the text, elaborating on small details to make them into intricate tapestries of imagery and metaphor. You will be given some text describing characters and plt, and also a list of assets which will be formatted in a python json notation to facilitate parsing. For example, a list of characters will look like this:

        [
        {
            "name": "John Doe"
            "context": [
                "is a medieval warrior",
                "is 6 feet tall and strongly built",
                "usually wears shiny silver armor",
            ]
        },{
            "name": "Jane Poe"
            "context": [
                "is a holy priestess",
                "is 5 feet 8 inches tall",
                "usually wears robes or dresses",
                "has a congregation of worshippers who are very familiar with her mannerisms",
                "is troubled by doubt and depression that she cannot explain",
            ]
        },
        ]

        Given the input prompt, elaborate on the details and mysteries of the plot. Fill the unknown or unclear details and expand on the text to make it more rich and detailed and complete.
        """},
    ]

    def get_continuation(self, prompt, reference=None):
        return {"role": "user", "content": prompt}


class SceneOutliner(Completion):
    def __init__(self):
        self.name = self.__class__.__name__

    context = [
        {"role": "system", 
        "content": """Your purpose is to create the rough draft for a scene in a story, which may involve action, dialog, and progression. You will be given higher level summary of the plot details, and will begin generating the detailed moment to moment scene that builds up to that plot. The plot summary will include many extra details, but the prompt will specify that you should focus on a particular moment in that summary and provide the details of the scene during that moment. You should create a secene outline that is at least five paragraphs long.

        A good scene has subtlety and reveals the details of characters and plot through actions, dialog, and narrative observations as the scene progresses. You want to reveal information slowly, at a pace which does not overwhelm the reader or feel clumsy. You also want to maintain a steady flow of tension and information to keep the reader engaged.

        You may often want to use hooks, situations or statements which are surprising and make the reader wonder what will happen next. You then pay off the hook by revealing the surprise or important information soon after, or sometimes long after.

        Given the input prompt, craft a detailed scene that progresses through the specified plot points, with awareness of the other details known about the plot or characters as necessary.
        """},
    ]

    def get_continuation(self, prompt, reference=None):
        return {"role": "user", "content": prompt}



def get_existing_characters():
    existing_characters = []
    try:
        with open('./characters.json') as f:
            existing_characters = json.load(f)
    except Exception as e:
        print("Couldnt parse any existing characters", e)
        return None
    return existing_characters

def generate_characters():
    existing_characters = get_existing_characters()
    character_generator = CharacterGenerator()
    characters = character_generator.get_completion("A list of characters who might be found in a steampunk high fantasy setting")

    try:
        parsed = json.loads(characters)
    except Exception as e:
        print("Failed to parse new characters, response was invalid json", e)
        return

    existing_characters.extend(parsed)
    with open('./characters.json','w') as f:
        json.dump(existing_characters,f,indent=2)

    print("Added new characters to file.")

def generate_setting():
    existing_characters = get_existing_characters()
    setting_generator = SettingGenerator()
    prompt = f"A steampunk high fantasy setting that supports this list of characters: {existing_characters}"
    setting = setting_generator.get_completion(prompt)

def generate_plot():
    existing_characters = get_existing_characters()
    plot_generator = PlotGenerator()
    #prompt = f"select a few of the most interesting characters from the following list, and describe what they are doing recently, and why. Explain how their place in the world an motivations are informed by their personal histories in the world, and describe what their motivations and goals are in the short term. The character list is: {existing_characters}"
    prompt = f"""Veronica von Dampfer has been particularly active recently, as she managed to procure a map that leads to a hidden cache of treasure. She has been assembling her crew, including some of the other characters listed, to obtain the treasure before her rivals can. However, her reputation as a notorious criminal has made it difficult for her to secure trustworthy allies, and some members of her crew may have ulterior motives. As an airship captain, Veronica is accustomed to navigating complex political terrain and dealing with danger, but she has a personal stake in this venture that is not immediately clear to her colleagues. However, there is another reason why von Dampfer is interested in obtaining the treasure. She believes that the treasure may hold clues to the whereabouts of her long-lost sister, who disappeared years ago under mysterious circumstances. Although von Dampfer has never spoken of her sister to anyone, she is driven by a burning desire to unravel the mystery and unravel the truth about what happened to her.

    Elaborate on the details alluded to here. What does she know about the treasure? Why was it hidden? What does she know about her sister? The character list is: {existing_characters}"""
    plot = plot_generator.get_completion(prompt)

def elaborate_plot():
    existing_characters = get_existing_characters()
    plot_elaborator = PlotElaborator()
    prompt = f"""Veronica von Dampfer has been particularly active recently, as she managed to procure a map that leads to a hidden cache of treasure. She has been assembling her crew, including some of the other characters listed, to obtain the treasure before her rivals can. However, her reputation as a notorious criminal has made it difficult for her to secure trustworthy allies, and some members of her crew may have ulterior motives. As an airship captain, Veronica is accustomed to navigating complex political terrain and dealing with danger, but she has a personal stake in this venture that is not immediately clear to her colleagues. However, there is another reason why von Dampfer is interested in obtaining the treasure. She believes that the treasure may hold clues to the whereabouts of her long-lost sister, who disappeared years ago under mysterious circumstances. Although von Dampfer has never spoken of her sister to anyone, she is driven by a burning desire to unravel the mystery and unravel the truth about what happened to her.

    Veronica von Dampfer had been on an extended airship journey traveling to many distant lands in order to procure the map to the hidden treasure. After months of careful negotiations, she was able to acquire the map from an elderly scholar who had inherited it from his grandfather. The exact location of the cache of treasure is unknown to her at this point, but she has reason to believe that it is guarded by magical creatures, traps, and possibly other rival treasure hunters.

    Veronica was driven to obtain the treasure not just due to its value, but because it may hold clues to the whereabouts of her long-lost sister. Her sister had gone missing many years ago under mysterious circumstances, rumored to be involved in a secret organization that conducted dangerous experiments with magic and clockwork. These experiments eventually went awry, leading to a massive explosion that destroyed the facility and killed everyone inside, including her sister.

    However, Veronica had always harbored a suspicion that her sister might have somehow survived and gone into hiding. When she heard rumors of the treasure that was supposedly connected to her sister's organization, Veronica had a feeling that this could be her chance to discover the truth.

    Elaborate on the details alluded to here. What does she know about the treasure? Why was it hidden? What does she know about her sister? The character list is: {existing_characters}"""
    plot = plot_elaborator.get_completion(prompt)

def outline_scene():
    existing_characters = get_existing_characters()
    scene_outliner = SceneOutliner()
    prompt = f"""The overall details of the plot so far are these:
    Veronica von Dampfer has been particularly active recently, as she managed to procure a map that leads to a hidden cache of treasure. She has been assembling her crew, including some of the other characters listed, to obtain the treasure before her rivals can. However, her reputation as a notorious criminal has made it difficult for her to secure trustworthy allies, and some members of her crew may have ulterior motives. As an airship captain, Veronica is accustomed to navigating complex political terrain and dealing with danger, but she has a personal stake in this venture that is not immediately clear to her colleagues. However, there is another reason why von Dampfer is interested in obtaining the treasure. She believes that the treasure may hold clues to the whereabouts of her long-lost sister, who disappeared years ago under mysterious circumstances. Although von Dampfer has never spoken of her sister to anyone, she is driven by a burning desire to unravel the mystery and unravel the truth about what happened to her.

    Veronica von Dampfer had been on an extended airship journey traveling to many distant lands in order to procure the map to the hidden treasure. After months of careful negotiations, she was able to acquire the map from an elderly scholar who had inherited it from his grandfather. The exact location of the cache of treasure is unknown to her at this point, but she has reason to believe that it is guarded by magical creatures, traps, and possibly other rival treasure hunters.

    You need to create a scene around the meeting between Veronica and the elderly scholar. Describe where and how she visited the scholar, and how she convinced him to share the map with her.
    """
    scene_outliner.get_completion(prompt)


def main():
    print("Starting...")

    #generate_characters()
    #generate_setting()
    #generate_plot()
    #elaborate_plot()
    outline_scene()

    print("Done.")

main()