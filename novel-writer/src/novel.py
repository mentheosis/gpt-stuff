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
            messages=messages,
            temperature=0.8,
        )
        end = time.perf_counter()
        print(f"Res after {end - start} seconds:\n",res.choices[0].message.content)
        print("Usage", res.usage )
        return res.choices[0].message.content

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
        "content": """Your purpose is to create the outline for a scene in a story, which may involve action, dialog, and progression. You will be given higher level summary of the plot details, and will begin generating the detailed moment to moment scene that builds up to that plot. The plot summary will include many extra details, but the prompt will specify that you should focus on a particular moment in that summary and provide the details of the scene during that moment. You should create a secene outline that is at least five paragraphs long.

        A good scene has subtlety and reveals the details of characters and plot through actions, dialog, and narrative observations as the scene progresses. You want to reveal information slowly, at a pace which does not overwhelm the reader or feel clumsy. You also want to maintain a steady flow of tension and information to keep the reader engaged.

        You may often want to use hooks, situations or statements which are surprising and make the reader wonder what will happen next. You then pay off the hook by revealing the surprise or important information soon after, or sometimes long after.

        Here is an example of an outline of a scene derived from a short plot summary. The plot summary is: 
        'Traveling through Missouri, Joel and Ellie are forced to take a detour through Kansas City, where they are ambushed. Joel kills two of the bandits, but a third overpowers him and nearly chokes him to death before Ellie saves him by shooting the man with Frank's pistol. More bandits find the bodies; their leader, Kathleen, believes Joel and Ellie might be in contact with a man named Henry and orders a manhunt. Joel counsels Ellie about the firefight and gives her the pistol back. Kathleen's second-in-command Perry thinks he has found Henry's hideout, but something is growing under the building. Kathleen orders it kept secret until they find Henry.' 

        The outline of the scene from that plot summary is:
        'Camping in the woods for the night, Joel warns Ellie not to trust anyone they meet. The next day, they reach the ruins of Kansas City, Missouri. The highway is blocked, forcing them to take a detour into the city. Ellie sees a man begging for help, but Joel drives at him. A brick breaks the truck's window, a spike strip punctures the tires, and gunfire sends the pair careening into a laundromat.

        Ellie hides as Joel kills two men with a rifle. A third—Bryan (Juan Magana)—gets the drop on him. As Joel is being choked, Ellie takes the handgun and shoots Bryan in the back, paralyzing him from the waist down. Joel confiscates her gun and sends her away before fatally stabbing Bryan as he screams for mercy. Joel and Ellie escape as more bandits—part of a group that overthrew the government and took control of the city—find the bodies. Their leader, Kathleen Coghlan (Melanie Lynskey), is informed of the events. She openly postulates her enemies—including Henry Burrell, who she believes ratted out her brother to be executed—are responsible for contacting the killers and orders her followers to search the city. Meanwhile, Joel teaches Ellie how to properly hold her gun and agrees to let her carry it.

        Kathleen's second-in-command, Perry (Jeffrey Pierce), shows her a vacated room where Henry had been living. The floor of the basement-level storage room is buckling, and underground something is moving. Perry insists they deal with the problem, but Kathleen orders him to hide the evidence until they find Henry. Joel locates a high-rise building where they can get a good view of the surrounding area and find an escape route. Lying down to sleep in one of the apartments, one of Ellie's jokes makes Joel laugh for the first time. Abruptly awoken by Ellie's voice, Joel finds Henry (Lamar Johnson) and his eight-year-old brother Sam (Keivonn Montreal Woodard) holding them at gunpoint.'

        Now, given the input prompt, craft a detailed scene that progresses through the specified plot points, with awareness of the other details known about the plot or characters as necessary.
        """},
    ]

    def get_continuation(self, prompt, reference=None):
        return {"role": "user", "content": prompt}


class SceneDetailer(Completion):
    def __init__(self):
        self.name = self.__class__.__name__

    context = [
        {"role": "system", 
        "content": """ You are a novelist who writes dense and florid prose. You receive a very basic summary of plot points as an outline, and then produce a rich and dense literary exposition describing the scenes summarized by the outline. 

        You avoid repeating any phrase that is exactly copied from the summary, instead you always bring a new original way of describing things. You introduce ideas in a subtle way at first, and then return to the idea with more detail a few sentences later. Before introducing characters or setting, you use several sentences to prepare their introduction and describe some details setting a stage for them. You space out the points from the summary, adding additional entertaining details or diversions between them.

        Here are some example of how to transform summaries into florid prose:

        {Summary prompt: 
The narrator of "Bartleby the Scrivener" is the Lawyer, who runs a law practice on Wall Street in New York. While the Lawyer knows many interesting stories of scriveners, he bypasses them all in favor of telling the story of Bartleby, whom he finds to be the most interesting. Before introducing Bartleby, the Lawyer describes the other scriveners working in his office at this time. The first is Turkey, who becomes flushed with an ill temper in the afternoon. The second worker is Nippers, who is much younger and more ambitious than Turkey. The last employee is not a scrivener, but an errand-boy named Ginger Nut. The Lawyer spends some time describing the habits of these men and then introduces Bartleby. 

        Prose from the summary:
I am a rather elderly man. The nature of my avocations for the last thirty years has brought me into more than ordinary contact with what would seem an interesting and somewhat singular set of men, of whom as yet nothing that I know of has ever been written:—I mean the law-copyists or scriveners. I have known very many of them, professionally and privately, and if I pleased, could relate divers histories, at which good-natured gentlemen might smile, and sentimental souls might weep. But I waive the biographies of all other scriveners for a few passages in the life of Bartleby, who was a scrivener of the strangest I ever saw or heard of. While of other law-copyists I might write the complete life, of Bartleby nothing of that sort can be done. I believe that no materials exist for a full and satisfactory biography of this man. It is an irreparable loss to literature. Bartleby was one of those beings of whom nothing is ascertainable, except from the original sources, and in his case those are very small. What my own astonished eyes saw of Bartleby, that is all I know of him, except, indeed, one vague report which will appear in the sequel.

Ere introducing the scrivener, as he first appeared to me, it is fit I make some mention of myself, my employés, my business, my chambers, and general surroundings; because some such description is indispensable to an adequate understanding of the chief character about to be presented.

My chambers were up stairs at No.—Wall-street. At one end they looked upon the white wall of the interior of a spacious sky-light shaft, penetrating the building from top to bottom. This view might have been considered rather tame than otherwise, deficient in what landscape painters call “life.” But if so, the view from the other end of my chambers offered, at least, a contrast, if nothing more. In that direction my windows commanded an unobstructed view of a lofty brick wall, black by age and everlasting shade; which wall required no spy-glass to bring out its lurking beauties, but for the benefit of all near-sighted spectators, was pushed up to within ten feet of my window panes. Owing to the great height of the surrounding buildings, and my chambers being on the second floor, the interval between this wall and mine not a little resembled a huge square cistern.

At the period just preceding the advent of Bartleby, I had two persons as copyists in my employment, and a promising lad as an office-boy. First, Turkey; second, Nippers; third, Ginger Nut. These may seem names, the like of which are not usually found in the Directory. In truth they were nicknames, mutually conferred upon each other by my three clerks, and were deemed expressive of their respective persons or characters. Turkey was a short, pursy Englishman of about my own age, that is, somewhere not far from sixty. In the morning, one might say, his face was of a fine florid hue, but after twelve o’clock, meridian—his dinner hour—it blazed like a grate full of Christmas coals; and continued blazing—but, as it were, with a gradual wane—till 6 o’clock, P.M. or thereabouts, after which I saw no more of the proprietor of the face, which gaining its meridian with the sun, seemed to set with it, to rise, culminate, and decline the following day, with the like regularity and undiminished glory. 

Nippers, the second on my list, was a whiskered, sallow, and, upon the whole, rather piratical-looking young man of about five and twenty. I always deemed him the victim of two evil powers—ambition and indigestion.

Ginger Nut, the third on my list, was a lad some twelve years old. His father was a carman, ambitious of seeing his son on the bench instead of a cart, before he died. So he sent him to my office as student at law, errand boy, and cleaner and sweeper, at the rate of one dollar a week.}

        {Summary prompt: 
Bartleby comes to the office to answer an ad placed by the Lawyer, who at that time needed more help. The Lawyer hires Bartleby and gives him a space in the office. At first, Bartleby seems to be an excellent worker. One day, the Lawyer has a small document he needs examined. He calls Bartleby in to do the job, but Bartleby responds: "I would prefer not to." This answer amazes the Lawyer. He is so amazed by this response, and the calm way Bartleby says it, that he cannot even bring himself to scold Bartleby. Instead, he calls in Nippers to examine the document instead.

        Prose from the summary:
Now my original business—that of a conveyancer and title hunter, and drawer-up of recondite documents of all sorts—was considerably increased by receiving the master’s office. There was now great work for scriveners. Not only must I push the clerks already with me, but I must have additional help. In answer to my advertisement, a motionless young man one morning, stood upon my office threshold, the door being open, for it was summer. I can see that figure now—pallidly neat, pitiably respectable, incurably forlorn! It was Bartleby.

After a few words touching his qualifications, I engaged him, glad to have among my corps of copyists a man of so singularly sedate an aspect, which I thought might operate beneficially upon the flighty temper of Turkey, and the fiery one of Nippers.

I should have stated before that ground glass folding-doors divided my premises into two parts, one of which was occupied by my scriveners, the other by myself. According to my humor I threw open these doors, or closed them. I resolved to assign Bartleby a corner by the folding-doors, but on my side of them, so as to have this quiet man within easy call, in case any trifling thing was to be done. I placed his desk close up to a small side-window in that part of the room, a window which originally had afforded a lateral view of certain grimy back-yards and bricks, but which, owing to subsequent erections, commanded at present no view at all, though it gave some light. Within three feet of the panes was a wall, and the light came down from far above, between two lofty buildings, as from a very small opening in a dome. Still further to a satisfactory arrangement, I procured a high green folding screen, which might entirely isolate Bartleby from my sight, though not remove him from my voice. And thus, in a manner, privacy and society were conjoined.

At first Bartleby did an extraordinary quantity of writing. As if long famishing for something to copy, he seemed to gorge himself on my documents. There was no pause for digestion. He ran a day and night line, copying by sun-light and by candle-light. I should have been quite delighted with his application, had he been cheerfully industrious. But he wrote on silently, palely, mechanically.

Now and then, in the haste of business, it had been my habit to assist in comparing some brief document myself, calling Turkey or Nippers for this purpose. One object I had in placing Bartleby so handy to me behind the screen, was to avail myself of his services on such trivial occasions. It was on the third day, I think, of his being with me, and before any necessity had arisen for having his own writing examined, that, being much hurried to complete a small affair I had in hand, I abruptly called to Bartleby. In my haste and natural expectancy of instant compliance, I sat with my head bent over the original on my desk, and my right hand sideways, and somewhat nervously extended with the copy, so that immediately upon emerging from his retreat, Bartleby might snatch it and proceed to business without the least delay.

In this very attitude did I sit when I called to him, rapidly stating what it was I wanted him to do—namely, to examine a small paper with me. Imagine my surprise, nay, my consternation, when without moving from his privacy, Bartleby in a singularly mild, firm voice, replied, “I would prefer not to.”

I sat awhile in perfect silence, rallying my stunned faculties. Immediately it occurred to me that my ears had deceived me, or Bartleby had entirely misunderstood my meaning. I repeated my request in the clearest tone I could assume. But in quite as clear a one came the previous reply, “I would prefer not to.”

“Prefer not to,” echoed I, rising in high excitement, and crossing the room with a stride. “What do you mean? Are you moon-struck? I want you to help me compare this sheet here—take it,” and I thrust it towards him.

“I would prefer not to,” said he.

I looked at him steadfastly. His face was leanly composed; his gray eye dimly calm. Not a wrinkle of agitation rippled him. Had there been the least uneasiness, anger, impatience or impertinence in his manner; in other words, had there been any thing ordinarily human about him, doubtless I should have violently dismissed him from the premises. But as it was, I should have as soon thought of turning my pale plaster-of-paris bust of Cicero out of doors. I stood gazing at him awhile, as he went on with his own writing, and then reseated myself at my desk. This is very strange, thought I. What had one best do? But my business hurried me. I concluded to forget the matter for the present, reserving it for my future leisure. So calling Nippers from the other room, the paper was speedily examined.}
    """}]

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
    #existing_characters = get_existing_characters()
    scene_outliner = SceneOutliner()
    prompt = f"""The overall summary of the plot so far is this:
    'Veronica von Dampfer has been particularly active recently, as she managed to procure a map that leads to a hidden cache of treasure. She has been assembling her crew, including some of the other characters listed, to obtain the treasure before her rivals can. However, her reputation as a notorious criminal has made it difficult for her to secure trustworthy allies, and some members of her crew may have ulterior motives. As an airship captain, Veronica is accustomed to navigating complex political terrain and dealing with danger, but she has a personal stake in this venture that is not immediately clear to her colleagues. However, there is another reason why von Dampfer is interested in obtaining the treasure. She believes that the treasure may hold clues to the whereabouts of her long-lost sister, who disappeared years ago under mysterious circumstances. Although von Dampfer has never spoken of her sister to anyone, she is driven by a burning desire to unravel the mystery and unravel the truth about what happened to her.

    Veronica von Dampfer had been on an extended airship journey traveling to many distant lands in order to procure the map to the hidden treasure. After months of careful negotiations, she was able to acquire the map from an elderly scholar who had inherited it from his grandfather. The exact location of the cache of treasure is unknown to her at this point, but she has reason to believe that it is guarded by magical creatures, traps, and possibly other rival treasure hunters.'

    You need to outline the scene around the meeting between Veronica and the elderly scholar. Describe where and how she found the scholar, and how the interaction went when she met him and convinced him to share the map with her.
    """
    return scene_outliner.get_completion(prompt)

def detail_scene(prompt):
    scene_detailer = SceneDetailer()
    scene_detailer.get_completion(prompt)


def main():
    print("Starting...")

    #generate_characters()
    #generate_setting()
    #generate_plot()
    #elaborate_plot()
    #outline = outline_scene()

    outline = """ Veronica von Dampfer traveled to a remote and squalid town on the edge of a vast desert to meet with the elderly scholar. She found him holed up in a ramshackle inn, surrounded by dusty books and vials of strange liquids. The scholar was suspicious at first, eyeing Veronica up and down with a look of disapproval, but she managed to win him over with her persuasive charms.

Veronica sat at the scholar's table and made small talk, trying to gauge his interests and figure out what would convince him to part with the map. She noticed that he was particularly proud of his collection of ancient artifacts, and she began to speak about her own interests in history and archaeology, dropping hints about her connections to academic circles and her wealth as a successful airship captain.

As the evening wore on, Veronica turned the conversation to the map, asking the scholar if he knew anything about it. He hesitated at first, telling her that it was a family heirloom and that he couldn't simply give it away. However, Veronica pressed on, telling him that she was willing to pay handsomely for the map, and that she would make sure that it was put to good use.

Eventually, the scholar relented, and he went to retrieve the map from a dusty shelf in the corner of the room. He unrolled it carefully and explained to Veronica what each of the symbols and markings meant. Veronica listened carefully, committing every detail to memory, and asked a few questions to clarify certain points.

When it was time for her to leave, Veronica thanked the scholar warmly and slipped a small pouch of coins over to him as a token of her appreciation. He seemed genuinely touched, and he bowed to Veronica before she headed out the door.

As she stepped out into the dusty streets, Veronica was feeling elated. She knew that the journey ahead would be fraught with danger, but she couldn't help but feel a sense of excitement at the prospect of discovering the cache of treasure and, perhaps, some answers about her missing sister.
    """

    split_outline = outline.split(".")
    for item in split_outline:
        if item != '':
            prompt=f"Here is the summary: '{item}' - Now craft a detailed scene of florid prose from the summary."
            print("\n\nPrompt:", prompt)
            detail_scene(prompt)

    # prompt=f"Here is the summary: '{outline}' - Now craft a detailed scene of florid prose from the summary."
    # detail_scene(prompt)

    print("\n\nDone.")

main()