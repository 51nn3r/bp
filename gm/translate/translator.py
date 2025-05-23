from transformers import pipeline


def get_translator():
    return pipeline("translation", model="facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="slk_Latn")
