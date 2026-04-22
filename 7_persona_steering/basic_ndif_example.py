from nnsight.modeling.language import LanguageModel

import os


model = LanguageModel("meta-llama/Llama-3.3-70B-Instruct")
# model = LanguageModel("meta-llama/Llama-3.1-8B-Instruct")


with model.trace("The Eiffel Tower is in the city of", remote=True):
    output = model.output.save()



