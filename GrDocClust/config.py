import GrDocClust.algos as algos
import GrDocClust.metrics as metrics
import GrDocClust.utils as utils

# Config values. 
ROOT_DIR = 'C:\\Users\\g.georgariou\\Documents\\Visual Studio Code Projects\\GreekDocumentClustering\\'
# ROOT_DIR = 'PATH_TO\\GreekDocumentClustering\\'
csv_dir = f'{ROOT_DIR}Results\\' 
figures_dir = f'{ROOT_DIR}figures\\'
local_datasets_path = f'{ROOT_DIR}Datasets\\'
local_models_storage_path = f'E:\\NLP_Models_Cached_Local\\NLP_Models_Cached_Local\\'
#local_models_storage_path = f'{ROOT_DIR}NLP_Models_Cached_Local\\'
local_precomputed_vectors_path = f'{ROOT_DIR}Precomputed_Vectors\\'

# ------------------------ Datasets - Corpus ------------------------ #
datasets_strings = [
    "test",
    #"greeksum",
    #"greek_reddit",
    #"pubmed4000_greek",
]

def datasets_pointers():
    return {
        "test": utils.load_dataset_test,
        "greeksum": utils.load_dataset_greeksum,
        "greek_reddit": utils.load_dataset_greek_reddit,
        "pubmed4000_greek": utils.load_dataset_pubmed4000_greek,
    }

# ------------------------ Embeddings - Doc Vectors ------------------------ #
vectorizers_strings = [
    "tfidf",

    "greek_spacy_model_embeddings" ,

    "greek_bert_model_embeddings",
    "st_greek_media_model_embeddings",
    "jina_v3_model_embeddings",
    "sent_transformers_paraph_multi_model_embeddings",
    "greek_xlm_roberta_model_embeddings",
]

def vectorizers_pointers():
    return {
        "tfidf": utils.tfidf,

        "greek_spacy_model_embeddings": utils.spacy_model_embeddings,

        "greek_bert_model_embeddings": utils.sent_transformers_model_embeddings,
        "st_greek_media_model_embeddings": utils.sent_transformers_model_embeddings,
        "jina_v3_model_embeddings": utils.sent_transformers_model_embeddings,
        "sent_transformers_paraph_multi_model_embeddings": utils.sent_transformers_model_embeddings,
        "greek_xlm_roberta_model_embeddings": utils.sent_transformers_model_embeddings,
        
    }


# ------------------------ Clustering Algorithms ------------------------ #
clustering_algorithms_strings = [
    "kmeans",
    "kmedoids",
]

# Config Clustering algorithm approaches
def clustering_algorithms_parameteres():
    return {
        "kmeans": 
            ['n_clusters', 'algorithm', 'init_centers'],
        "kmedoids":
            ['n_clusters', 'method', 'init_centers'],
    }

def clustering_algorithms_arguments(n_clusters):
    return {
        "kmeans": [
            [n_clusters, 'elkan', 'random'],
        ],
        "kmedoids":[
            [n_clusters, 'pam', 'k-medoids++'],
        ],
    }
        
def clustering_algorithms_pointers():
    return {
        "kmeans": algos.kmeans,
        "kmedoids": algos.kmedoids,
    }



# ------------------------ Ext Evaluation Metrics ------------------------ #
evaluation_metrics_strings = [
    "accuracy",
    "adjusted_mutual_information",
    "fowlkes_mallows_index",
    "v_measure_index",
]

def evaluation_metrics_pointers():
    return {
        "accuracy": metrics.accuracy,
        "adjusted_mutual_information": metrics.adjusted_mutual_information,
        "fowlkes_mallows_index": metrics.fowlkes_mallows_index,
        "v_measure_index": metrics.v_measure_index,
    }


greek_stop_words = set(
"""
αδιάκοπα αι ακόμα ακόμη ακριβώς άλλα αλλά αλλαχού άλλες άλλη άλλην
άλλης αλλιώς αλλιώτικα άλλο άλλοι αλλοιώς αλλοιώτικα άλλον άλλος άλλοτε αλλού
άλλους άλλων άμα άμεσα αμέσως αν ανά ανάμεσα αναμεταξύ άνευ αντί αντίπερα αντίς
άνω ανωτέρω άξαφνα απ απέναντι από απόψε άρα άραγε αρκετά αρκετές
αρχικά ας αύριο αυτά αυτές αυτή αυτήν αυτής αυτό αυτοί αυτόν αυτός αυτού αυτούς
αυτών αφότου αφού

βέβαια βεβαιότατα

γι για γιατί γρήγορα γύρω

δα δε δείνα δεν δεξιά δήθεν δηλαδή δι δια διαρκώς δικά δικό δικοί δικός δικού
δικούς διόλου δίπλα δίχως

εάν εαυτό εαυτόν εαυτού εαυτούς εαυτών έγκαιρα εγκαίρως εγώ εδώ ειδεμή είθε είμαι
είμαστε είναι εις είσαι είσαστε είστε είτε είχα είχαμε είχαν είχατε είχε είχες έκαστα
έκαστες έκαστη έκαστην έκαστης έκαστο έκαστοι έκαστον έκαστος εκάστου εκάστους εκάστων
εκεί εκείνα εκείνες εκείνη εκείνην εκείνης εκείνο εκείνοι εκείνον εκείνος εκείνου
εκείνους εκείνων εκτός εμάς εμείς εμένα εμπρός εν ένα έναν ένας ενός εντελώς εντός
εναντίον  εξής  εξαιτίας  επιπλέον επόμενη εντωμεταξύ ενώ εξ έξαφνα εξήσ εξίσου έξω επάνω
επειδή έπειτα επί επίσης επομένως εσάς εσείς εσένα έστω εσύ ετέρα ετέραι ετέρας έτερες
έτερη έτερης έτερο έτεροι έτερον έτερος ετέρου έτερους ετέρων ετούτα ετούτες ετούτη ετούτην
ετούτης ετούτο ετούτοι ετούτον ετούτος ετούτου ετούτους ετούτων έτσι εύγε ευθύς ευτυχώς εφεξής
έχει έχεις έχετε έχομε έχουμε έχουν εχτές έχω έως έγιναν  έγινε  έκανε  έξι  έχοντας

η ήδη ήμασταν ήμαστε ήμουν ήσασταν ήσαστε ήσουν ήταν ήτανε ήτοι ήττον

θα

ι ιδία ίδια ίδιαν ιδίας ίδιες ίδιο ίδιοι ίδιον ίδιοσ ίδιος ιδίου ίδιους ίδιων ιδίως ιι ιιι
ίσαμε ίσια ίσως

κάθε καθεμία καθεμίας καθένα καθένας καθενός καθετί καθόλου καθώς και κακά κακώς καλά
καλώς καμία καμίαν καμίας κάμποσα κάμποσες κάμποση κάμποσην κάμποσης κάμποσο κάμποσοι
κάμποσον κάμποσος κάμποσου κάμποσους κάμποσων κανείς κάνεν κανένα κανέναν κανένας
κανενός κάποια κάποιαν κάποιας κάποιες κάποιο κάποιοι κάποιον κάποιος κάποιου κάποιους
κάποιων κάποτε κάπου κάπως κατ κατά κάτι κατιτί κατόπιν κάτω κιόλας κλπ κοντά κτλ κυρίως

λιγάκι λίγο λιγότερο λόγω λοιπά λοιπόν

μα μαζί μακάρι μακρυά μάλιστα μάλλον μας με μεθαύριο μείον μέλει μέλλεται μεμιάς μεν
μερικά μερικές μερικοί μερικούς μερικών μέσα μετ μετά μεταξύ μέχρι μη μήδε μην μήπως
μήτε μια μιαν μιας μόλις μολονότι μονάχα μόνες μόνη μόνην μόνης μόνο μόνοι μονομιάς
μόνος μόνου μόνους μόνων μου μπορεί μπορούν μπρος μέσω  μία  μεσώ

να ναι νωρίς

ξανά ξαφνικά

ο οι όλα όλες όλη όλην όλης όλο ολόγυρα όλοι όλον ολονέν όλος ολότελα όλου όλους όλων
όλως ολωσδιόλου όμως όποια οποιαδήποτε οποίαν οποιανδήποτε οποίας οποίος οποιασδήποτε οποιδήποτε
όποιες οποιεσδήποτε όποιο οποιοδηήποτε όποιοι όποιον οποιονδήποτε όποιος οποιοσδήποτε
οποίου οποιουδήποτε οποίους οποιουσδήποτε οποίων οποιωνδήποτε όποτε οποτεδήποτε όπου
οπουδήποτε όπως ορισμένα ορισμένες ορισμένων ορισμένως όσα οσαδήποτε όσες οσεσδήποτε
όση οσηδήποτε όσην οσηνδήποτε όσης οσησδήποτε όσο οσοδήποτε όσοι οσοιδήποτε όσον οσονδήποτε
όσος οσοσδήποτε όσου οσουδήποτε όσους οσουσδήποτε όσων οσωνδήποτε όταν ότι οτιδήποτε
ότου ου ουδέ ούτε όχι οποία  οποίες  οποίο  οποίοι  οπότε  ος

πάνω  παρά  περί  πολλά  πολλές  πολλοί  πολλούς  που  πρώτα  πρώτες  πρώτη  πρώτο  πρώτος  πως
πάλι πάντα πάντοτε παντού πάντως πάρα πέρα πέρι περίπου περισσότερο πέρσι πέρυσι πια πιθανόν
πιο πίσω πλάι πλέον πλην ποιά ποιάν ποιάς ποιές ποιό ποιοί ποιόν ποιός ποιού ποιούς
ποιών πολύ πόσες πόση πόσην πόσης πόσοι πόσος πόσους πότε ποτέ πού πούθε πουθενά πρέπει
πριν προ προκειμένου πρόκειται πρόπερσι προς προτού προχθές προχτές πρωτύτερα πώς

σαν σας σε σεις σου στα στη στην στης στις στο στον στου στους στων συγχρόνως
συν συνάμα συνεπώς συχνάς συχνές συχνή συχνήν συχνής συχνό συχνοί συχνόν
συχνός συχνού συχνούς συχνών συχνώς σχεδόν

τα τάδε ταύτα ταύτες ταύτη ταύτην ταύτης ταύτοταύτον ταύτος ταύτου ταύτων τάχα τάχατε
τελευταία  τελευταίο  τελευταίος  τού  τρία  τρίτη  τρεις τελικά τελικώς τες τέτοια τέτοιαν
τέτοιας τέτοιες τέτοιο τέτοιοι τέτοιον τέτοιος τέτοιου
τέτοιους τέτοιων τη την της τι τίποτα τίποτε τις το τοι τον τοσ τόσα τόσες τόση τόσην
τόσης τόσο τόσοι τόσον τόσος τόσου τόσους τόσων τότε του τουλάχιστο τουλάχιστον τους τούς τούτα
τούτες τούτη τούτην τούτης τούτο τούτοι τούτοις τούτον τούτος τούτου τούτους τούτων τυχόν
των τώρα

υπ υπέρ υπό υπόψη υπόψιν ύστερα

χωρίς χωριστά

ω ως ωσάν ωσότου ώσπου ώστε ωστόσο ωχ
""".split()
)