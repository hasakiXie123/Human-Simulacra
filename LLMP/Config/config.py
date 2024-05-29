# path
Current_Directory = "/root/Desktop/LLMP"
LLM_Directory = "/root/Desktop/LLMP/LLMs"
Attributes_Directory = "/root/Desktop/LLMP/Characters/Attributes"
Profiles_Path = "/root/Desktop/LLMP/Characters/character_profiles.json"
Introductions_Path = "/root/Desktop/LLMP/Characters/character_introductions.json"
Story_Directory = "/root/Desktop/LLMP/Characters/Stories"
Memory_Directory = "/root/Desktop/LLMP/Characters/Memories"
Question_Directory = "/root/Desktop/LLMP/Characters/Questions"
OpenCompass_Directory = "/root/Desktop/LLMP/opencompass"
Output_Directory = "/root/Desktop/LLMP/Outputs"
Continued_story_path = "/root/Desktop/LLMP/Characters/Stories/Erica Walker/Erica Walker-19.txt" ## for continued story generation, if you want use this, please set the argument-Continued to True


# argument
Model_for_data = "gpt-3.5-turbo" # model for data generation
Context_length = 128000 # the context length for data generation model
Iteration_for_story = 10
Iteration_threshold = 50 # score threshold
Window_size = 5 # if story length > Context_length, context = [-Window_size, Window_size]
Continued = False ## for continued story generation
Method_list = ['none', 'base_prompt', 'base_rag', 'cognitive']
Character_list = ["Mary Jones", "Haley Collins", "Sara Ochoa", "James Jones", "Tami Clark", "Michael Miller", "Kevin Kelly", "Erica Walker", "Leslie Nichols", "Robert Scott", "Marsh Zhaleh"]


Model_for_evaluation = "gpt-4-1106-preview" 
Model_for_agent = "gpt-4-1106-preview" ## agent in MACM

# when evaluate remote model
OPENAI_API_KEY = "Your OPENAI_API_KEY"
OPENAI_BASE_URL = "https://api.openai.com/v1"
API_KEY = "Your API_KEY"
BASE_URL = "https://api.openai.com/v1" # or any api company

# when evaluate local model
# OPENAI_API_KEY = "EMPTY"
# OPENAI_BASE_URL = "http://localhost:8000/v1" 
# API_KEY = "EMPTY"
# BASE_URL = "http://localhost:8000/v1" 







