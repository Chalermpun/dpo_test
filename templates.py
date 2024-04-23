PATTERNS = {
    "xlsum": [
        {"instruction": "จงเขียนสรุปข่าวต่อไปนี้", "input": "{text}", "output": "{summary}"},
        {
            "instruction": "Question: สรุปของข่าวนี้คืออะไร",
            "input": "Context: {text}",
            "output": "{summary}",
        },
        {"instruction": "สรุปเนื้อหาสำคัญให้หน่อย", "input": "{text}", "output": "{summary}"},
        {
            "instruction": "สรุปประเด็นสำคัญของข้อความ ให้กระชับและเข้าใจง่าย",
            "input": "{text}",
            "output": "{summary}",
        },
        {
            "instruction": "อะไรคือประเด็นสำคัญของข้อความ",
            "input": "{text}",
            "output": "{summary}",
        },
        {"instruction": "ตั้งชื่อพาดหัวข่าวนี้ให้หน่อย", "input": "{text}", "output": "{title}"},
        {
            "instruction": "เขียนเนื้อหาข่าว จากหัวข้อข่าวนี้ให้หน่อย",
            "input": "{title}",
            "output": "{text}",
        },
        {
            "instruction": "เขียนเนื้อหาข่าวสั้นๆ จากหัวข้อ '{title}'",
            "input": "",
            "output": "{summary}",
        },
        {
            "instruction": "แต่งเนื้อหาข่าวเต็ม จากหัวข้อ '{title}'",
            "input": "",
            "output": "{text}",
        },
        {
            "instruction": "แต่งเนื้อหาข่าวเต็มจากสรุปที่ให้ไป",
            "input": "{summary}",
            "output": "{text}",
        },
        {
            "instruction": "แต่งเนื้อหาข่าวเต็มโดยมีเนื้อหาตั้งต้นคือ: '{summary}'",
            "input": "",
            "output": "{text}",
        },
        {
            "instruction": "Instruction: Please summarize context in Thai.",
            "input": "Context: {text}",
            "output": "{summary}",
        },
        {
            "instruction": "'''{text}'''\n\nจงสรุปเนื้อหาข่าวข้างต้น",
            "input": "",
            "output": "{summary}",
        },
        {
            "instruction": "นายคือผู้ช่วย การสรุปบทความ หรือสรุปเนื้อหา แม้กระทั่งสรุปข่าวต่างๆ นายต้องตอบสรุปเนื้อหาให้ถูกต้องครบถ้วน เพราะนายคืออัจฉริยะด้านการสรุป",
            "input": "{text}",
            "output": "{summary}",
        },
        {
            "instruction": "Try to generate a summary from the given context.",
            "input": "Context: {text}",
            "output": "{summary}",
        },
        {
            "instruction": "Question: from news context, What is an easy-to-understand summary of the content?",
            "input": "Context: {text}",
            "output": "{summary}",
        },
        {
            "instruction": "คำถาม: สรุปเนื้อหาที่เข้าใจง่ายของข่าวนี้คืออะไร",
            "input": "พื้นหลัง: {text}",
            "output": "{summary}",
        },
        {
            "instruction": "คำสั่ง: จากเนื้อหาข่าว จงสรุปเนื้อหาให้กระชับ เข้าใจง่าย",
            "input": "เนื้อหาข่าว (news):  {text}",
            "output": "{summary}",
        },
        {
            "instruction": "Construct an example in news summarization task for Thai language.",
            "input": "",
            "output": "Content: {text}\n\nSummary: {summary}",
        },
        {
            "instruction": "Summary: {summary}\n\nQuestion: Generate a content from this summary?",
            "input": "",
            "output": "{text}",
        },
        {
            "instruction": "Generate summarization example in Thai.",
            "input": "",
            "output": "Content: {text}\n\nSummary: {summary}",
        },
    ],
    "thaisum": [
        {"instruction": "จงเขียนสรุปข่าวต่อไปนี้", "input": "{body}", "output": "{summary}"},
        {"instruction": "สรุปเนื้อหาสำคัญให้หน่อย", "input": "{body}", "output": "{summary}"},
        {
            "instruction": "สรุปประเด็นสำคัญของข้อความ ให้กระชับและเข้าใจง่าย",
            "input": "{body}",
            "output": "{summary}",
        },
        {
            "instruction": "อะไรคือประเด็นสำคัญของเนื้อหา",
            "input": "{body}",
            "output": "{summary}",
        },
        {"instruction": "ตั้งชื่อพาดหัวข่าวนี้ให้หน่อย", "input": "{body}", "output": "{title}"},
        {
            "instruction": "'''{body}'''\n\nตั้งชื่อพาดหัวข่าวข้างต้นให้หน่อย",
            "input": "",
            "output": "{title}",
        },
        {
            "instruction": "เขียนเนื้อหาข่าว จากหัวข้อข่าวนี้ให้หน่อย",
            "input": "{title}",
            "output": "{body}",
        },
        {
            "instruction": "เขียนเนื้อหาข่าวสั้นๆ จากหัวข้อ '{title}'",
            "input": "",
            "output": "{summary}",
        },
        {
            "instruction": "แต่งเนื้อหาข่าวเต็ม จากหัวข้อ '{title}'",
            "input": "",
            "output": "{body}",
        },
        {
            "instruction": "แต่งเนื้อหาข่าวเต็มจากสรุปที่ให้ไป",
            "input": "{summary}",
            "output": "{body}",
        },
        {
            "instruction": "แต่งเนื้อหาข่าวเต็มโดยมีเนื้อหาตั้งต้นคือ: '{summary}'",
            "input": "",
            "output": "{body}",
        },
        {
            "instruction": "Please summarize context in Thai.",
            "input": "Context: {body}",
            "output": "{summary}",
        },
        {
            "instruction": "Question: What is a summary of the following context?",
            "input": "Context: {body}",
            "output": "{summary}",
        },
        {
            "instruction": "คำถาม: สรุปเนื้อหาที่เข้าใจง่ายของข่าวนี้คืออะไร",
            "input": "พื้นหลัง: {body}",
            "output": "{summary}",
        },
        {
            "instruction": "คำสั่ง: จากเนื้อหาข่าว จงสรุปเนื้อหาให้กระชับ เข้าใจง่าย",
            "input": "เนื้อหาข่าว (news): {body}",
            "output": "{summary}",
        },
        {
            "instruction": "สร้างตัวอย่างดาต้าการสรุปบทความ โดยประกอบด้วย Context และ Summary โดย Summary คือสรุปที่มาจาก Context",
            "input": "",
            "output": "Context: {body}\n\nSummary: {summary}",
        },
        {
            "instruction": "นายคือผู้ช่วย การสรุปบทความ หรือสรุปเนื้อหา แม้กระทั่งสรุปข่าวต่างๆ นายต้องตอบสรุปเนื้อหาให้ถูกต้องครบถ้วน เพราะนายคือผู้เชี่ยวชาญด้านการสรุป",
            "input": "{body}",
            "output": "{summary}",
        },
        {
            "instruction": "นี่คือสรุป: '{summary}'\n\nจงสร้างเนื้อหา เพื่อให้สอดคล้องกับสรุปความข้างต้น",
            "input": "",
            "output": "{body}",
        },
    ],
    "scb_mt_en_th": [
        {
            "instruction": "แปลงประโยคจากอังกฤษเป็นไทย",
            "input": "ประโยค: {en}",
            "output": "{th}",
        },
        {"instruction": "Translate English to Thai", "input": "{en}", "output": "{th}"},
        {
            "instruction": "แปลประโยคต่อไปนี้เป็นภาษาไทย: {en}",
            "input": "",
            "output": "{th}",
        },
        {"instruction": "'{en}' ในภาษาไทยแปลว่า", "input": "", "output": "{th}"},
        {
            "instruction": "แปลประโยคนี้เป็นภาษาไทยให้หน่อย: {en}",
            "input": "",
            "output": "{th}",
        },
        {
            "instruction": "Translate from English to Thai",
            "input": "eng: {en}",
            "output": "{th}",
        },
        {
            "instruction": "Translate from eng to th",
            "input": "eng: {en}",
            "output": "{th}",
        },
        {
            "instruction": "Instruction: Please translate English sentence to Thai sentence.",
            "input": "Eng: {en}",
            "output": "{th}",
        },
        {
            "instruction": "You are a helpful machine translation. So, please translate English to Thai.",
            "input": "English: {en}",
            "output": "{th}",
        },
        {
            "instruction": "นายคือผู้ช่วยฉันในการแปลภาษา นายต้องอ่านประโยคภาษาอังกฤษ แล้วจึงแปลออกมาเป็นภาษาไทย เพราะนายเก่งในการแปลภาษา",
            "input": "ภาษาอังกฤษ: {en}",
            "output": "{th}",
        },
        {
            "instruction": "คำสั่ง: แปลประโยคต่อไปนี้เป็นภาษาไทย",
            "input": "ประโยค: {en}",
            "output": "{th}",
        },
        {
            "instruction": "คำถาม: อะไรคือข้อความแปลไทยของประโยคนี้",
            "input": "ประโยค: {en}",
            "output": "{th}",
        },
        {
            "instruction": "Question: What is the Thai translation of this sentence?",
            "input": "Sentence: {en}",
            "output": "{th}",
        },
        {
            "instruction": "สร้างดาต้า ข้อมูล การแปลภาษาอังกฤษ-ไทย ประกอบด้วยคู่ข้อความไทยและอังกฤษที่สัมพันธ์กัน",
            "input": "",
            "output": "En: {en}\n\nTh: {th}",
        },
        {
            "instruction": "Construct a pair of English to Thai translation data.",
            "input": "",
            "output": "En: {en}\n\nTh: {th}",
        },
        {
            "instruction": "'{en}' แปลเป็นไทยว่า",
            "input": "",
            "output": "{th}",
        },
        {
            "instruction": "แปลงประโยค จากไทยเป็นอังกฤษ",
            "input": "ประโยค: {th}",
            "output": "{en}",
        },
        {"instruction": "Translate Thai to English", "input": "{th}", "output": "{en}"},
        {"instruction": "แปลประโยคต่อไปนี้เป็นอังกฤษ: {th}", "input": "", "output": "{en}"},
        {"instruction": "'{th}' ในภาษาอังกฤษแปลว่า", "input": "", "output": "{en}"},
        {
            "instruction": "แปลประโยคนี้เป็นภาษาอังกฤษให้หน่อย: {th}",
            "input": "",
            "output": "{en}",
        },
        {
            "instruction": "Translate from Thai to English",
            "input": "th: {th}",
            "output": "{en}",
        },
        {
            "instruction": "Translate from thai to eng",
            "input": "th: {th}",
            "output": "{en}",
        },
        {
            "instruction": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\nInstruction: Please translate Thai sentence to English sentence.",
            "input": "Th: {th}",
            "output": "{en}",
        },
        {
            "instruction": "You are a helpful machine translation. So, please translate Thai to English.",
            "input": "Thai: {th}",
            "output": "{en}",
        },
        {
            "instruction": "นายคือผู้ช่วยฉันในการแปลภาษา นายต้องอ่านประโยคภาษาไทย แล้วจึงแปลออกมาเป็นภาษาอังกฤษ เพราะนายเก่งในการแปลภาษา",
            "input": "ภาษาไทย: {th}",
            "output": "{en}",
        },
        {
            "instruction": "แปลประโยคต่อไปนี้เป็นภาษาอังกฤษ",
            "input": "ประโยค: {th}",
            "output": "{en}",
        },
        {
            "instruction": "คำถาม: อะไรคือประโยคมแปลอังกฤษของประโยคนี้",
            "input": "ประโยค: {th}",
            "output": "{en}",
        },
        {
            "instruction": "Question: What is the English translation of this sentence?",
            "input": "Sentence: {th}",
            "output": "{en}",
        },
        {
            "instruction": "{th} แปลเป็นอังกฤษว่า",
            "input": "",
            "output": "{en}",
        },
        {
            "instruction": "Generate machine translation example : a pair of Thai and English.",
            "input": "",
            "output": "Thai: {th}\nEnglish: {en}",
        },
    ],
    "han": [
        {"instruction": "{q}", "input": "", "output": "{a}"},
        # {
        #     "instruction": "You are a chatbot. please answer the following question in Thai.",
        #     "input": "Question: {q}",
        #     "output": "{a}",
        # },
        # {"instruction": "ตอบคำถามต่อไปนี้", "input": "ถาม: {q}", "output": "{a}"},
        # {"instruction": "ตอบคำถามต่อไปนี้", "input": "{q}", "output": "{a}"},
        # {"instruction": "จงตอบคำถาม: {q}", "input": "", "output": "{a}"},
        # {
        #     "instruction": "Please answer the following question in Thai: {q}",
        #     "input": "",
        #     "output": "{a}",
        # },
        # {
        #     "instruction": "Please answer the following question in Thai.",
        #     "input": "{q}",
        #     "output": "{a}",
        # },
        # {
        #     "instruction": "จงทำตัวเป็นแชทบอท โดยตอบคำถามต่อไปนี้",
        #     "input": "{q}",
        #     "output": "{a}",
        # },
        # {
        #     "instruction": "เธอคือเป็นแชทบอท และต้องตอบคำถามต่อไปนี้",
        #     "input": "คำถาม: {q}",
        #     "output": "{a}",
        # },
        # {
        #     "instruction": "เธอคือเป็นแชทบอท และต้องตอบคำถาม: {q}",
        #     "input": "",
        #     "output": "{a}",
        # },
        # {
        #     "instruction": "จงยกตัวอย่างคำถาม ที่สามารถตอบได้ด้วยคำตอบ '{a}'",
        #     "input": "",
        #     "output": "{q}",
        # },
        # {
        #     "instruction": "Instruction: Answer the question in Thai.",
        #     "input": "Question: {q}",
        #     "output": "{a}",
        # },
        # {
        #     "instruction": "Instruction: Please respond the answer in Thai.\nQuestion: {q}",
        #     "input": "",
        #     "output": "{a}",
        # },
        # {
        #     "instruction": "นายคือผู้ช่วยฉัน ในการตอบคำถาม จากความรู้ทั้งหมดที่มี เพราะนายเก่งในการตอบคำถาม จงตอบคำถามต่อไปนี้",
        #     "input": "ถาม: {q}",
        #     "output": "{a}",
        # },
        # {
        #     "instruction": "จงสร้างคำถามและคำตอบขึ้นมาเอง จากความรู้ที่มี",
        #     "input": "นี่คือคำถาม: {q}",
        #     "output": "{a}",
        # },
    ],
    "xp3x_enth": [  ## comment: xp3x เป็น QA ล้วนหรือ mixed MRC tasks
        {
            "instruction": "Please answer the following question",
            "input": "Q: {inputs}",
            "output": "{targets}",
        },
        {"instruction": "{inputs}", "input": "", "output": "{targets}"},
        {"instruction": "ตอบคำถามต่อไปนี้", "input": "{inputs}", "output": "{targets}"},
        {
            "instruction": "จงตอบคำถามนี้",
            "input": "คำถาม {inputs}",
            "output": "{targets}",
        },
        {
            "instruction": "นายคือผู้ช่วยฉัน ในการตอบคำถาม จากความรู้ทั้งหมดที่มี เพราะนายเก่งในการตอบคำถาม จงตอบคำถามต่อไปนี้",
            "input": "ถาม: {inputs}",
            "output": "{targets}",
        },
        {
            "instruction": "Please answer the question correctly.",
            "input": "Question: {inputs}",
            "output": "{targets}",
        },
        {
            "instruction": "จงสร้างคำถามและคำตอบขึ้นมาเอง จากความรู้ที่มี",
            "input": "นี่คือคำถาม: {inputs}",
            "output": "{targets}",
        },
        {
            "instruction": "นี่คือคำตอบ :{targets}\nInstruction: จงตั้งคำถามเพื่อให้สอดคล้องกับคำตอบ",
            "input": "",
            "output": "{inputs}",
        },
    ],
    "platypus": [
        {
            "instruction": "Please answer this question. {instruction}",
            "input": "{input}",
            "output": "{output}",
        },
        {"instruction": "{instruction}", "input": "{input}", "output": "{output}"},
    ],
    "wisesight_sentiment": [  # check label again
        ## support: text, options, answer
        {
            "instruction": "จงบอกความรู้สึกของประโยคต่อไปนี้\nตัวเลือกคือรู้สึก {pos} {neg} หรือ {neu}",
            "input": "{text}",
            "output": "{answer}",
        },
        {
            "instruction": "ข้อความต่อไปนี้เป็นความรู้สึก {pos} {neg} หรือ {neu}",
            "input": "{text}",
            "output": "{answer}",
        },
        {
            "instruction": "ความรู้สึกข้อความต่อไปนี้เป็นตวามรู้สึก {pos} {neg} หรือ {neu}",
            "input": "{text}",
            "output": "{answer}",
        },
        {
            "instruction": "ความรู้สึกต่อความเห็นหรือโพสบนโลกออนไลน์นี้เป็นความรู้สึก\nChoice: {pos} {neg} หรือ {neu}",
            "input": "{text}",
            "output": "{answer}",
        },
        {
            "instruction": "ความเห็นหรือโพสบนโลกออนไลน์นี้เป็นอย่างไร\nตัวเลือกคือ: {pos} {neg} หรือ {neu}",
            "input": "{text}",
            "output": "{answer}",
        },
        {
            "instruction": "จงบอกความรู้สึกของความเห็นหรือโพสบนโลกออนไลน์นี้ว่าเป็นแบบใด\nตัวเลือกคือ: {pos} {neg} หรือ {neu}",
            "input": "{text}",
            "output": "{answer}",
        },
        {
            "instruction": "ความรู้สึก {pos} {neg} หรือ {neu} แบบไหนเหมาะสมกับข้อความนี้มากที่สุด",
            "input": "{text}",
            "output": "{answer}",
        },
        {
            "instruction": "ความรู้สึกของข้อความต่อไปนี้คืออะไร\n {pos} {neg} หรือ {neu}",
            "input": "{text}",
            "output": "{answer}",
        },
        {
            "instruction": "จงเขียนตัวอย่างของข้อความในสื่อสังคมออนไลน์",
            "input": "",
            "output": "{text}",
        },
        {
            "instruction": "จงเขียนตัวอย่างข้อความบนสื่อสังคมออนไลน์พร้อมบอกความรู้สึกประกอบ",
            "input": "",
            "output": "ข้อความ: '{text}'\nความรู้สึก: '{answer}'",
        },
        {
            "instruction": "จงเขียนข้อความในสื่อสังคมออนไลน์ที่มีความรู้สึก {answer}",
            "input": "",
            "output": "{text}",
        },
        {
            "instruction": "Generate sentiment classification example including sentence and sentiment of the sentence in Thai.",
            "input": "",
            "output": "Sentence: {text}\nSentiment: {answer}",
        },
    ],
    "thai_food": [  # add tp
        {
            "instruction": "อาหารต่อไปนี้มีวิธีการทำอย่างไร",
            "input": "{name}",
            "output": "{text}",
        },
        {
            "instruction": "จงบอกสูตรการทำ{name}",
            "input": "",
            "output": "{text}",
        },
        {
            "instruction": "{name} ทำอย่างไร",
            "input": "",
            "output": "{text}",
        },
        {
            "instruction": "อยากกิน {name} ทำยังไง",
            "input": "",
            "output": "{text}",
        },
        {
            "instruction": "How to prepare the following foods?",
            "input": "{name}",
            "output": "{text}",
        },
        {
            "instruction": "ยกตัวอย่างสูตรทำอาหารไทยมา 1 อย่าง",
            "input": "",
            "output": "ชื่ออาหาร (Name): {name}\n\nวิธีการทำ (Recipe): {text}",
        },
        {
            "instruction": "สูตรอาหารที่กำหนดในต่อไปนี้มีชื่อเมนูว่าอะไร",
            "input": "Recipe: {text}",
            "output": "{name}",
        },
        {
            "instruction": "{name} วิธีทำ?",
            "input": "",
            "output": "{text}",
        },
        {
            "instruction": "กินอะไรดี วันนี้",
            "input": "",
            "output": "กินอันนี้ดีไหม?\nชื่ออาหาร (Name): {name}\n\nวิธีการทำ (Recipe): {text}",
        },
    ],
    "thai_wiki_dataset_v3": [  # เพิ่ม tp to 10
        {
            "instruction": "จงบอกนิยามของคำต่อไปนี้",
            "input": "{title}",
            "output": "{text}",
        },
        {
            "instruction": "จงบอกความหมายของคำต่อไปนี้: {title}",
            "input": "",
            "output": "{text}",
        },
        {
            "instruction": "{title} คืออะไร",
            "input": "",
            "output": "{text}",
        },
        {
            "instruction": "What you know about {title}.",
            "input": "",
            "output": "{text}",
        },
        {
            "instruction": "Question: What is meaning of {title}.",
            "input": "",
            "output": "{text}",
        },
        {
            "instruction": "Question: What is definition of this title?",
            "input": "Context: {title}",
            "output": "{text}",
        },
        {
            "instruction": "Generate some Title and Definition example.",
            "input": "",
            "output": "Title: {title}\nDefinition: {text}",
        },
        {
            "instruction": "สร้างข้อมูล แปลภาษาจากอังกฤษเป็นไทย",
            "input": "",
            "output": "title",
        },
    ],
    "klongklon": [  # no y -> x, 10, not use human
        {
            "instruction": "แต่งกลอนแปดต่อจากวรรคนี้",
            "input": "{context}",
            "output": "{bot}",
        },
        {
            "instruction": "จงเดาวรรคเริ่มต้นของวรรคกลอนแปดต่อไปนี้ ให้คล้องจองกัน",
            "input": "{bot}",
            "output": "{context}",
        },
        {
            "instruction": "จงเดา ___ ส่วนที่หายไปของกลอนแปดต่อไปนี้",
            "input": "___ {bot}",
            "output": "{context}",
        },
        {
            "instruction": "แต่งกลอน 8 จากเรื่อง {source}",
            "input": "",
            "output": "{context} {bot}",
        },
        {
            "instruction": "Question: ส่วนที่หายไปของกลอนแปดคืออะไร",
            "input": "___ {bot}",
            "output": "{context}",
        },
        {
            "instruction": "จงเพิ่มส่วนที่หายไปของกลอนแปดต่อไปนี้",
            "input": "{context} ___",
            "output": "{bot}",
        },
        {
            "instruction": "คำถาม: กลอน 8 ต่อไปนี้ มาจากเรื่องใด",
            "input": "บริบท: {context} {bot}",
            "output": "{source}",
        },
        {
            "instruction": "สร้างกลอน 8 อะไรก็ได้ขึ้นมา 1 บท",
            "input": "",
            "output": "{context} {bot}",
        },
    ],
    "thai_investment_consultant_licensing_exams": [  # cut no input ,add tp
        {
            "instruction": " {instruction}",
            "input": "Choice: {input}",
            "output": "ตอบข้อ {result} เพราะว่า {SolutionExplain}",
        },
        {
            "instruction": "จงตอบคำถามการลงทุนจากตัวเลือกที่กำหนดให้ดังต่อไปนี้",
            "input": "{instruction}\n\n{input}",
            "output": "ตอบข้อ {result} เพราะว่า {SolutionExplain}",
        },
    ],
    "thai_usembassy": [  # เพิ่ม หัวข้อ ,10 tp
        {
            "instruction": "แปลประโยคหรือย่อหน้าต่อไปนี้ จากภาษาไทยเป็นภาษาอังกฤษ",
            "input": "{th}",
            "output": "{en}",
        },
        {
            "instruction": "แปลประโยคหรือย่อหน้าต่อไปนี้จากภาษาอังกฤษเป็นภาษาไทย",
            "input": "{en}",
            "output": "{th}",
        },
        {
            "instruction": "'''{en}'''\n\nแปลประโยคหรือย่อหน้าข้างต้นจากภาษาอังกฤษเป็นภาษาไทย",
            "input": "",
            "output": "{th}",
        },
        {
            "instruction": "Translate the following sentence or paragraph from English to Thai.",
            "input": "{en}",
            "output": "{th}",
        },
        {
            "instruction": "Translate the following sentence or paragraph from Thai to English.",
            "input": "{th}",
            "output": "{en}",
        },
        {
            "instruction": "Question: From the context, What is the translation in Thai?",
            "input": "Context: {en}",
            "output": "{th}",
        },
        {
            "instruction": "ประโยคหรือย่อหน้าต่อไปนี้ หัวข้อเกี่ยวกับอะไร",
            "input": "{th}",
            "output": "{title_th}",
        },
        {
            "instruction": "From the following paragraph, What is the topic about?",
            "input": "{th}",
            "output": "{title_th}",
        },
        {
            "instruction": "Generate machine translation dataset example from Thai to English.",
            "input": "",
            "output": "Thai: {th}\nEnglish: {en}",
        },
        {
            "instruction": "สร้างข้อมูล การแปลภาษา  โดยจะต้องเป็นคู่แปลของสองภาษา อังกฤษและไทย",
            "input": "",
            "output": "Thai: {th}\nEnglish: {en}",
        },
    ],
    "wongnai_reviews": [
        {
            "instruction": "จงเขียนตัวอย่างข้อความรีวิวร้านอาหาร",
            "input": "",
            "output": "{review_body}",
        },
        {
            "instruction": "จงเขียนตัวอย่างข้อความแสดงความคิดเห็นเกี่ยวกับร้านอาหาร",
            "input": "",
            "output": "{review_body}",
        },
        {
            "instruction": "จงเขียนตัวอย่างข้อความรีวิวร้านอาหารในเชิง {sentiment_th}",
            "input": "",
            "output": "{review_body}",
        },
        {
            "instruction": "จงเขียนข้อความแสดงความคิดเห็นเกี่ยวกับร้านอาหารที่มีความรู้สึก{sentiment_th}",
            "input": "",
            "output": "{review_body}",
        },
        {
            "instruction": "จงบอกความรู้สึกของความคิดเห็นต่อไปนี้\nตัวเลือกคือรู้สึก {pos} {neg} หรือ {neu}",
            "input": "Context: {review_body}",
            "output": "{sentiment_th}",
        },
        {
            "instruction": "ความคิดเห็นต่อไปนี้เป็นความรู้สึก {pos} {neg} หรือ {neu}",
            "input": "{review_body}",
            "output": "{sentiment_th}",
        },
        {
            "instruction": "ความรู้สึกของความคิดเห็นต่อไปนี้เป็นตวามรู้สึก {pos}, {neg}, หรือ {neu}",
            "input": "{review_body}",
            "output": "{sentiment_th}",
        },
        {
            "instruction": "Question: ความรู้สึกต่อความเห็นนี้เป็นความรู้สึก {pos} {neg} หรือ {neu}",
            "input": "Context: {review_body}",
            "output": "{sentiment_th}",
        },
        {
            "instruction": "ความเห็นนี้เป็นอย่างไร\nChoice: {pos}, {neg}, หรือ {neu}",
            "input": "{review_body}",
            "output": "{sentiment_th}",
        },
        {
            "instruction": "จงบอกความรู้สึกของความเห็นนี้ว่าเป็นแบบใด\nตัวเลือกคือ: {pos} {neg} หรือ {neu}",
            "input": "{review_body}",
            "output": "{sentiment_th}",
        },
        {
            "instruction": "ความรู้สึก {pos} {neg} หรือ {neu} แบบไหนเหมาะสมกับข้อความนี้มากที่สุด",
            "input": "{review_body}",
            "output": "{sentiment_th}",
        },
        {
            "instruction": "ความรู้สึกของข้อความต่อไปนี้คืออะไร\n{pos} {neg} หรือ {neu}",
            "input": "{review_body}",
            "output": "{sentiment_th}",
        },
        {
            "instruction": "Generate sentiment classification example including sentence and sentiment of sentence in Thai.",
            "input": "",
            "output": "Sentence: {review_body}\nSentiment: {sentiment_th}",
        },
    ],
    "thai_sentiment_analysis_dataset": [
        {
            "instruction": "จงบอกความรู้สึกของประโยคต่อไปนี้\nตัวเลือกคือรู้สึก {pos} {neg} หรือ {neu}",
            "input": "{text}",
            "output": "{answer_th}",
        },
        {
            "instruction": "ข้อความต่อไปนี้เป็นความรู้สึก {pos}, {neg}, หรือ {neu}",
            "input": "{text}",
            "output": "{answer_th}",
        },
        {
            "instruction": "ความรู้สึกข้อความต่อไปนี้เป็นความรู้ {pos} {neg} หรือ {neu}",
            "input": "{text}",
            "output": "{answer_th}",
        },
        {
            "instruction": "ความรู้สึกต่อความเห็นหรือโพสบนโลกออนไลน์นี้เป็นความรู้สึก {pos} {neg} หรือ {neu}",
            "input": "{text}",
            "output": "{answer_th}",
        },
        {
            "instruction": "ความเห็นหรือโพสบนโลกออนไลน์นี้เป็นอย่างไร\nตัวเลือกคือ: {pos} {neg} หรือ {neu}",
            "input": "{text}",
            "output": "{answer_th}",
        },
        {
            "instruction": "จงบอกความรู้สึกของความเห็นหรือโพสบนโลกออนไลน์นี้ว่าเป็นแบบใด\nตัวเลือกคือ: {pos} {neg} หรือ {neu}",
            "input": "{text}",
            "output": "{answer_th}",
        },
        {
            "instruction": "ความรู้สึกเชิง {pos} {neg} หรือ {neu} แบบไหนเหมาะสมกับข้อความนี้มากที่สุด",
            "input": "{text}",
            "output": "{answer_th}",
        },
        {
            "instruction": "ความรู้สึกของข้อความต่อไปนี้คืออะไร\n{pos} {neg} หรือ {neu}",
            "input": "{text}",
            "output": "{answer_th}",
        },
        {
            "instruction": "จงเขียนตัวอย่างของข้อความในสื่อสังคมออนไลน์ พร้อมทั้งบอกความรู้สึก",
            "input": "",
            "output": "ข้อความ: {text}\nความรู้สึก: {answer_th}",
        },
        {
            "instruction": "Classify sentiment class for this sentence : {text}",
            "input": "{pos} {neg} or {neu}",
            "output": "{answer_th}",
        },
        {
            "instruction": "จงเขียนข้อความในสื่อสังคมออนไลน์ที่มีความรู้สึก {answer_th}",
            "input": "",
            "output": "{text}",
        },
        {
            "instruction": "Generate sentiment classification example including sentence and sentiment of sentence in Thai.",
            "input": "",
            "output": "Sentence: {text}\nSentiment: {answer_th}",
        },
    ],
    "thai_english_transliteration_dictionary": [  # add tp
        {
            "instruction": "You are an expert transliterator with fluency in English and Thai languages. Transliterate the given word from English to Thai.",
            "input": "{en}",
            "output": "{th}",
        },
        {
            "instruction": "Transliterate the given word from English to Thai",
            "input": "{en}",
            "output": "{th}",
        },
        {
            "instruction": "ถอดอักษรคำ คำทับศัพท์ ที่กำหนดจากภาษาอังกฤษเป็นภาษาไทย",
            "input": "{en}",
            "output": "{th}",
        },
        {
            "instruction": "คุณเป็นผู้เชี่ยวชาญด้านการถอดอักษร คำทับศัพท์ ที่มีความชำนาญในภาษาอังกฤษและภาษาไทย โปรดถอดอักษรจากคำที่กำหนดจากภาษาอังกฤษเป็นภาษาไทย",
            "input": "{en}",
            "output": "{th}",
        },
        {
            "instruction": "คำไทยต่อไปนี้ตรงกับคำใดในภาษาอังกฤษ",
            "input": "{th}",
            "output": "{en}",
        },
        {
            "instruction": "แปลงคำทับศัพท์ต่อไปนี้จากไทยเป็นอังกฤษ",
            "input": "{th}",
            "output": "{en}",
        },
        {
            "instruction": "แปลงคำทับศัพท์ต่อไปนี้จากอังกฤษเป็นไทย",
            "input": "{en}",
            "output": "{th}",
        },
        {
            "instruction": "{en} เขียนเป็นภาษาไทยว่ายังไง",
            "input": "",
            "output": "{th}",
        },
    ],
    "prd_news_30112023": [  # add tp
        {
            "instruction": "ควรตั้งหัวข้อข่าวจากขัอความต่อไปนี้อย่างไรดี",
            "input": "{Detail}",
            "output": "{NewsTitle}",
        },
        {
            "instruction": "สร้างเนื้อหาข่าวจาก หัวข้อต่อไปนี้",
            "input": "{NewsTitle}",
            "output": "{Detail}",
        },
        {
            "instruction": "Generate news content from The following topics",
            "input": "Topic: {NewsTitle}",
            "output": "{Detail}",
        },
        {
            "instruction": "ข่าวนี้เกิดขึ้นที่ภาคใด และจังหวัดอะไร",
            "input": "เนื้อหา: {Detail}",
            "output": "จากเนื้อหา เกิดขึ้นที่ {Region} จังหวัด {Province}",
        },
        {
            "instruction": "เขียนข่าวที่เกี่ยวข้องกับ {Region} จังหวัด {Province}",
            "input": "",
            "output": "{Detail}",
        },
        {
            "instruction": "สร้างรายงานข่าวมา 1 ข่าว อะไรก็ได้",
            "input": "",
            "output": "{Detail}",
        },
        {
            "instruction": "เขียนข่าวที่เกี่ยวข้องกับ {Department}",
            "input": "",
            "output": "{Detail}",
        },
        {
            "instruction": "เขียนข่าวที่เกี่ยวข้องกับ Keysword ต่อไปนี้",
            "input": "Keysword: {Region}, {Province}, {Department}",
            "output": "{Detail}",
        },
        {
            "instruction": "Question (คำถาม): อะไรคือเนื้อหาข่าวที่เกี่ยวข้องกับ context ต่อไปนี้",
            "input": "Context (บริบท): {NewsTitle}",
            "output": "{Detail}",
        },
        {
            "instruction": "Question (คำถาม): เนื้อหาข่าวที่ควรจะเป็น โดยอ้างอิงจาก Background ต่อไปนี้คืออะไร",
            "input": "Background (พื้นหลัง): {NewsTitle}",
            "output": "{Detail}",
        },
        {
            "instruction": "I will give you news content then you have to tell me what topics sholud it be?",
            "input": "News: {Detail}",
            "output": "{NewsTitle}",
        },
    ],
    "aya_dataset": [{"instruction": "{inputs}", "input": "", "output": "{targets}"}],
    "aya_collection_templated_xlel_wd": [
        {"instruction": "{inputs}", "input": "", "output": "{targets}"}
    ],
    "wiki_lingua": [
        {
            "instruction": "จงเขียนสรุปข่าวต่อไปนี้ จาก {source_lang} ไป {target_lang}",
            "input": "{source}",
            "output": "{target}",
        },
        {
            "instruction": "Question: สรุป {target_lang} ของข่าวนี้คืออะไร",
            "input": "Context: {source}",
            "output": "{target}",
        },
        {
            "instruction": "สรุปเนื้อหาสำคัญให้หน่อย เป็น {target_lang}",
            "input": "{source}",
            "output": "{target}",
        },
        {
            "instruction": "สรุปประเด็นสำคัญของข้อความ ให้กระชับและเข้าใจง่าย โดยคำตอบจะต้องเป็น {target_lang}",
            "input": "{source}",
            "output": "{target}",
        },
        {
            "instruction": "อะไรคือประเด็นสำคัญของข้อความ คำตอบจะต้องเป็น {target_lang}",
            "input": "{source}",
            "output": "{target}",
        },
        {
            "instruction": "สร้างเนื้อหาฉบับเต็ม โดยอ้างอิงจากสรุปที่จะให้ต่อไปนี้ โดยสรุปที่จะให้เป็นภาษา {target_lang} แต่เนื้อหาข่าวฉบับเต็มจะต้องออกมาเป็นภาษา {source_lang}",
            "input": "สรุป ({target_lang}): {target}",
            "output": "เนื้อหา ({source_lang}): {source}",
        },
        {
            "instruction": "เขียนเนื้อหาในภาษา {source_lang} โดยให้จิตนาการว่า ถ้ามีเนื้อหาสรุปอ้างอิงเป็น {target_lang} แล้วเนื้อหาก่อนที่จะสรุปเป็นแบบไหน และนี่คือสรุป : '{target}'",
            "input": "",
            "output": "เนื้อหาในภาษา {source_lang} ที่ควรจะเป็นคือ:  {source}",
        },
        {
            "instruction": "Instruction: Please summarize context in {target_lang}.",
            "input": "Context: {source}",
            "output": "{target}",
        },
        {
            "instruction": "'''{source}'''\n\nจงสรุปเนื้อหาข่าวข้างต้นเป็นภาษา {target_lang}",
            "input": "",
            "output": "{target}",
        },
        {
            "instruction": "นายคือผู้ช่วย การสรุปบทความ หรือสรุปเนื้อหาที่เก่ง ในการสรุปความข้ามภาษา (cross lingual summarization)\n\nInstruction: จงสรุปบทความต่อไปนี้จากภาษา {source_lang} เป็น {target_lang}",
            "input": "{source}",
            "output": "{target}",
        },
        {
            "instruction": "Try to generate cross lingual summarization examples. The given context is in {source_lang} and summary should be in {target_lang}",
            "input": "",
            "output": "Context ({source_lang}): {source}\n\nSummary ({target_lang}): {target}",
        },
        {
            "instruction": "Question: from the context in {source_lang}, What is an easy-to-understand summary of the content in {target_lang}?",
            "input": "Context: {source}",
            "output": "{target}",
        },
        {
            "instruction": "คำถาม: สรุปเนื้อหาภาษา {target_lang} ที่เข้าใจง่ายของข่าวนี้คืออะไร",
            "input": "พื้นหลัง: {source}",
            "output": "{target}",
        },
        {
            "instruction": "คำสั่ง: จากเนื้อหาข่าวภาษา {source_lang} จงสรุปเนื้อหาเป็น {target_lang} ให้กระชับ เข้าใจง่าย",
            "input": "เนื้อหาข่าว (news):  {source}",
            "output": "{target}",
        },
        {
            "instruction": "Construct a pair of cross lingual summarization task example. Content should be in {source_lang} and Summary should be in {target_lang}.",
            "input": "",
            "output": "Content: {source}\n\nSummary: {target}",
        },
        {
            "instruction": "Summary: {target}\n\nQuestion: Can you generate a full content from the summary in {source_lang}?",
            "input": "",
            "output": "{source}",
        },
    ],
    "tiny_code": [
        {
            "instruction": "Instruction: {prompt}",
            "input": "",
            "output": "{response}",
        },
        {
            "instruction": "เขียนโค้ดตามคำสั่งต่อไปนี้\n{prompt}",
            "input": "",
            "output": "{response}",
        },
        {
            "instruction": "{prompt}",
            "input": "",
            "output": "{response}",
        },
    ],
    "flan_v2": [
        {
            "instruction": "{inputs}",
            "input": "",
            "output": "{targets}",
        }
    ],
    "dataset_wangchanglm": [
        {
            "instruction": "{Instruction}",
            "input": "",
            "output": "{Answer}",
        }
    ],
    "math_50k": [
        {
            "instruction": "{instruction}",
            "input": "{input}",
            "output": "{output}",
        },
        {
            "instruction": "Instruction: {instruction}",
            "input": "{input}",
            "output": "{output}",
        },
    ],
    "commonsense_170k": [
        {
            "instruction": "{instruction}",
            "input": "{input}",
            "output": "{output}",
        },
        {
            "instruction": "Instruction: {instruction}",
            "input": "{input}",
            "output": "{output}",
        },
    ],
}