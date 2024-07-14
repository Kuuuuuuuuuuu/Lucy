from gtts import gTTS
from playsound import playsound

text = "안녕"
lang = 'ko'

tts = gTTS(text= text, lang = lang)

tts.save('tts.mp3')
playsound(tts)