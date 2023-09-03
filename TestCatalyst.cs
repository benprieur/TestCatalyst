using Catalyst;
using Catalyst.Models;

using Mosaik.Core;
using Version = Mosaik.Core.Version;

using System;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;

using Microsoft.Extensions.Logging;
//using Microsoft.Extensions.Logging.Console;


namespace TestCatalyst
{
    class Program
    {
        static async Task Main(string[] args)
        {
            // Premier exemple - infos grammaire
            await tokenization(); 

            // Deuxième exemple - thèmes LDA
            await testLDA(); 
        
            // Troisième exemple - détection des langues
            await detectionLangues();
        }

        static async Task tokenization()
        {
            Console.WriteLine(" ");
            Console.WriteLine("--- TOKENIZATION ---");
            Console.WriteLine(" ");

            Catalyst.Models.French.Register();
            Console.OutputEncoding = Encoding.UTF8;
 
            var nlp = await Pipeline.ForAsync(Language.French);

            string sentences = "Nous eûmes le bouquet de la foire ; ces messieurs étaient contents de leur voyage, et tout fut réglé dans deux jours. Et en route pour Coulommiers, où nous arrivâmes sans accidents.";
            var doc = new Document(sentences, Language.French);
            
            nlp.ProcessSingle(doc);            
            foreach (var sentence in doc)
            {
                foreach (var word in sentence)
                {
                    Console.WriteLine(word.POS + "\t" + word.Value);
                }
            }
        }

        static async Task testLDA()
        {
            Console.WriteLine(" ");
            Console.WriteLine("--- LDA ---");
            Console.WriteLine(" ");

            Console.OutputEncoding = Encoding.UTF8;

            Catalyst.Models.English.Register();
            Storage.Current = new DiskStorage("catalyst-models");

            var (train, test) = await Corpus.Reuters.GetAsync();
            var nlp = Pipeline.For(Language.English);

            var trainDocs = nlp.Process(train).ToArray();
            var testDocs = nlp.Process(test).ToArray();

            using (var lda = new LDA(Language.English, 0, "reuters-lda-programmez"))
            {
                lda.Data.NumberOfTopics = 5; 
                lda.Train(trainDocs, Environment.ProcessorCount);
                await lda.StoreAsync();
            }
            
            Dictionary<LDA.LDATopicDescription, int> themes = new Dictionary<LDA.LDATopicDescription, int>();
            using (var lda = await LDA.FromStoreAsync(Language.English, 0, "reuters-lda-programmez"))
            {
                foreach (var doc in testDocs)
                {
                    if (lda.TryPredict(doc, out var topics))
                    { 
                        int topScoreId = -1;
                        float topScore = 0;    
                        foreach (var element in topics)
                        {
                            int id = element.TopicID;
                            float score = element.Score;
                            if (topScore < score)
                            {
                                topScore = score;
                                topScoreId = id;
                            }
                        }
                        LDA.LDATopicDescription desc;
                        bool found = false;
                        lda.TryDescribeTopic(topScoreId, out desc);
                        foreach(KeyValuePair<LDA.LDATopicDescription, int> element in themes)
                        {
                            if (element.Key.TopicID == topScoreId)
                            {
                                themes[element.Key] = (int)element.Value + 1 ;
                                found = true;
                                break;
                            }
                        }
                        if (found == false)
                            themes.Add(desc, 1);
                        }
                    }
                }

            foreach(KeyValuePair<LDA.LDATopicDescription, int> element in themes)
            {
                Console.WriteLine("THEME => " + element.Key.ToString() + " " + element.Value.ToString() + " documents");
            }        
        }   

        static async Task detectionLangues()        
        {
            Console.WriteLine(" ");
            Console.WriteLine("--- LANGUAGE DETECTION ---");
            Console.WriteLine(" ");

            Console.OutputEncoding = Encoding.UTF8;

            var cld2LanguageDetector     = await LanguageDetector.FromStoreAsync(Language.Any, Version.Latest, "");

            foreach (var (lang, text) in Data.PhrasesLanguesInconnues)
            {
                var doc2 = new Document(text);
                cld2LanguageDetector.Process(doc2);

                Console.WriteLine(lang + " (à deviner) : " + doc2.Language);           
            }
   
        }
    }

    public static class Data
    {
     
        public static Dictionary<Language, string> PhrasesLanguesInconnues = new Dictionary<Language, string>()
        {
            [Language.English] = "The quick brown fox jumps over the lazy dog.",
            [Language.Armenian] = "Սոֆին, Արթուր Հեկտորը իմ սերն են.",
            [Language.Bulgarian] = "Ах чудна българска земьо, полюшвай цъфтящи жита.",
            [Language.Catalan] = "Jove xef, porti whisky amb quinze glaçons d’hidrogen, coi!",
            [Language.Croatian] = "Gojazni đačić s biciklom drži hmelj i finu vatu u džepu nošnje.",
            [Language.Czech] = "Nechť již hříšné saxofony ďáblů rozezvučí síň úděsnými tóny waltzu, tanga a quickstepu.",
            [Language.Danish] = "Quizdeltagerne spiste jordbær med fløde, mens cirkusklovnen Walther spillede på xylofon.",
            [Language.Esperanto] = "Laŭ Ludoviko Zamenhof bongustas freŝa ĉeĥa manĝaĵo kun spicoj.",
            [Language.Estonian] = "Põdur Zagrebi tšellomängija-följetonist Ciqo külmetas kehvas garaažis",
            [Language.Finnish] = "Hyvän lorun sangen pieneksi hyödyksi jäi suomen kirjaimet.",
            [Language.French] = "Portez ce vieux whisky au juge blond qui fume",
            [Language.German] = "Franz jagt im komplett verwahrlosten Taxi quer durch Bayern",
            [Language.Hebrew] = "דג סקרן שט בים מאוכזב ולפתע מצא חברה dg sqrn šṭ bjM mʾwkzb wlptʿ mṣʾ ḥbrh",
            [Language.Hindi] = "ऋषियों को सताने वाले दुष्ट राक्षसों के राजा रावण का सर्वनाश करने वाले विष्णुवतार भगवान श्रीराम, अयोध्या के महाराज दशरथ के बड़े सपुत्र थे।",
            [Language.Icelandic] = "Kæmi ný öxi hér, ykist þjófum nú bæði víl og ádrepa.",
            [Language.Italian] = "Quel vituperabile xenofobo zelante assaggia il whisky ed esclama: alleluja!",
            [Language.Japanese] = "いろはにほへと ちりぬるを わかよたれそ つねならむ うゐのおくやま けふこえて あさきゆめみし ゑひもせす（ん）",
            [Language.Korean] = "키스의 고유조건은 입술끼리 만나야 하고 특별한 기술은 필요치 않다.",
            [Language.Malay] = "അജവും ആനയും ഐരാവതവും ഗരുഡനും കഠോര സ്വരം പൊഴിക്കെ ഹാരവും ഒഢ്യാണവും ഫാലത്തില്‍ മഞ്ഞളും ഈറന്‍ കേശത്തില്‍ ഔഷധ എണ്ണയുമായി ഋതുമതിയും അനഘയും ഭൂനാഥയുമായ ഉമ ദുഃഖഛവിയോടെ ഇടതു പാദം ഏന്തി ങ്യേയാദൃശം നിര്‍ഝരിയിലെ ചിറ്റലകളെ ഓമനിക്കുമ്പോള്‍ ബാ‍ലയുടെ കണ്‍കളില്‍ നീര്‍ ഊര്‍ന്നു വിങ്ങി.",
            [Language.Mongolian] = "Щётканы фермд пийшин цувъя. Бөгж зогсч хэльюү.",
            [Language.Norwegian] = "Vår sære Zulu fra badeøya spilte jo whist og quickstep i min taxi.",
            [Language.Polish] = "Jeżu klątw, spłódź Finom część gry hańb!",
            [Language.Portuguese] = "Um pequeno jabuti xereta viu dez cegonhas felizes.",
            [Language.Spanish] = "José compró una vieja zampoña en Perú. Excusándose, Sofía tiró su whisky al desagüe de la banqueta.",
            [Language.Swedish] = "Flygande bäckasiner söka hwila på mjuka tuvor.",
            [Language.Thai] = "เป็นมนุษย์สุดประเสริฐเลิศคุณค่า กว่าบรรดาฝูงสัตว์เดรัจฉาน จงฝ่าฟันพัฒนาวิชาการ อย่าล้างผลาญฤๅเข่นฆ่าบีฑาใคร ไม่ถือโทษโกรธแช่งซัดฮึดฮัดด่า หัดอภัยเหมือนกีฬาอัชฌาสัย ปฏิบัติประพฤติกฎกำหนดใจ พูดจาให้จ๊ะๆ จ๋าๆ น่าฟังเอยฯ",
            [Language.Ukrainian] = "Жебракують філософи при ґанку церкви в Гадячі, ще й шатро їхнє п’яне знаємо.",
            [Language.Urdu] = "ٹھنڈ میں، ایک قحط زدہ گاؤں سے گذرتے وقت ایک چڑچڑے، باأثر و فارغ شخص کو بعض جل پری نما اژدہے نظر آئے۔",
            [Language.Yoruba] = "Ìwò̩fà ń yò̩ séji tó gbojúmó̩, ó hàn pákànpò̩ gan-an nis̩é̩ rè̩ bó dò̩la.",
            [Language.Welsh] = "Parciais fy jac codi baw hud llawn dŵr ger tŷ Mabon.",
        };
    }    
}
