from transformers import pipeline

translator = pipeline("translation", model="facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="slk_Latn")


def translate_text(text):
    return translator(text, max_length=512)


translated = translate_text(
    "Explore resources, tutorials, API docs, and dynamic examples to get the most out of OpenAI's developer.")

print(translated[0]['translation_text'])
