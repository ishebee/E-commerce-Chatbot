from semantic_router import Route, SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder
from sentence_transformers import SentenceTransformer

_ = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load sentence transformer model
encoder = HuggingFaceEncoder(
    name="sentence-transformers/all-MiniLM-L6-v2"
)

# Define routes
faq = Route(
    name='faq',
    utterances=[
        "What is the return policy of the products?",
        "Do I get discount with the HDFC credit card?",
        "How can I track my order?",
        "What payment methods are accepted?",
        "How long does it take to process a refund?",
    ]
)

sql = Route(
    name='sql',
    utterances=[
        "I want to buy Nike shoes that have 50% discount.",
        "Are there any shoes under Rs. 3000?",
        "Do you have formal shoes in size 9?",
        "Are there any Puma shoes on sale?",
        "What is the price of Puma running shoes?",
    ]
)

# ✅ Create a router with defined routes
router = SemanticRouter(routes=[], encoder=encoder)  # **NO NEED to call router.build()**
router.add(faq)
router.add(sql)
if __name__ == "__main__":
    print("Category:", router("What is your policy on defective product?").name)
    print("Category:", router("Pink Puma shoes in price range 5000 to 1000").name)
