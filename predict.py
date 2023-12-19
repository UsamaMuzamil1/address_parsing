import spacy
import fr_core_news_md
nlp = fr_core_news_md.load()
#nlp=spacy.load("output\models\model-best")

address_list=["“Marhaba Bakers & Nimco Allama Rasheed Turabi Road Block N North Nazimabad Karachi Sindh”,Allama Rasheed Turabi Road,Block N,North Nazimabad,Karachi,Sindh,Marhaba Bakers & Nimco",
              "“Jama Madina Masjid Shahrah e Faisal Link Road Habibaabad Quaidabad Karachi Sindh”,Shahrah e Faisal Link Road,Habibaabad,Quaidabad,Karachi,Sindh,Jama Madina Masjid",
              ]

# Checking predictions for the NER model
for address in address_list:
    doc=nlp(address)
    ent_list=[(ent.text, ent.label_) for ent in doc.ents]
    print("Address string -> "+address)
    print("Parsed address -> "+str(ent_list))
    print("******")



address="“City Glass Faizan Street Block E North Nazimabad Karachi Sindh”,Faizan Street,Block E,North Nazimabad,Karachi,Sindh,City Glass"
doc=nlp(address)
ent_list=[(ent.text, ent.label_) for ent in doc.ents]
print("Address string -> "+address)
print("Parsed address -> "+str(ent_list))

