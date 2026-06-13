"""
Configuration and constants for the Redrob Candidate Ranking System.
All JD-specific scoring parameters, skill mappings, company lists, and weights.
"""

from datetime import date

# ============================================================================
# Reference date for recency calculations
# ============================================================================
REFERENCE_DATE = date(2026, 6, 1)

# ============================================================================
# Title alignment scoring
# ============================================================================
# Map of normalized title keywords → fitness score for the Senior AI Engineer role.
# Matched via case-insensitive substring against the candidate's current_title.
# Order matters: first match wins, so more specific titles go first.

TITLE_TIERS = [
    # Tier 1: Perfect fit — AI/ML Engineering titles (score 0.9-1.0)
    (["senior ai engineer", "lead ai engineer", "staff ai engineer", "principal ai engineer"], 1.0),
    (["senior machine learning engineer", "lead machine learning engineer", "staff machine learning engineer", "lead ml engineer", "staff ml engineer", "principal ml engineer"], 1.0),
    (["senior ml engineer"], 1.0),
    (["ai engineer"], 0.95),
    (["machine learning engineer", "ml engineer"], 0.95),
    (["deep learning engineer"], 0.93),
    (["nlp engineer", "nlu engineer", "natural language"], 0.93),
    (["search engineer", "retrieval engineer", "ranking engineer", "recommendation engineer"], 0.92),
    (["applied scientist", "applied ml scientist"], 0.90),

    # Tier 2: Strong fit — Data Science / Research with production (0.7-0.89)
    (["senior data scientist", "lead data scientist", "staff data scientist"], 0.85),
    (["data scientist"], 0.80),
    (["research engineer", "research scientist"], 0.75),
    (["mlops engineer", "ml platform engineer", "ml infrastructure"], 0.78),

    # Tier 3: Adjacent engineering roles (0.4-0.65)
    (["senior backend engineer", "senior software engineer", "senior swe"], 0.60),
    (["backend engineer", "software engineer", "swe"], 0.52),
    (["senior data engineer"], 0.55),
    (["data engineer"], 0.48),
    (["platform engineer", "infrastructure engineer"], 0.45),
    (["full stack", "fullstack"], 0.40),
    (["tech lead", "technical lead", "engineering lead"], 0.45),

    # Tier 4: Peripherally related (0.2-0.39)
    (["devops", "sre", "site reliability"], 0.30),
    (["frontend", "front-end", "ui engineer"], 0.22),
    (["qa engineer", "test engineer", "sdet"], 0.18),
    (["engineering manager"], 0.30),
    (["product manager"], 0.12),
    (["business analyst"], 0.10),
    (["project manager"], 0.08),

    # Tier 5: Anti-fit — keyword-stuffer traps (0.01-0.07)
    (["marketing manager", "marketing lead", "marketing director"], 0.02),
    (["hr manager", "hr lead", "human resources"], 0.02),
    (["accountant", "accounting", "finance manager"], 0.02),
    (["sales executive", "sales manager", "sales lead", "sales representative"], 0.02),
    (["customer support", "customer service", "support engineer", "support specialist"], 0.03),
    (["content writer", "content strategist", "content manager", "copywriter"], 0.04),
    (["graphic designer", "ui designer", "ux designer", "visual designer"], 0.03),
    (["civil engineer", "structural engineer"], 0.02),
    (["mechanical engineer", "manufacturing engineer"], 0.03),
    (["operations manager", "operations lead"], 0.04),
    (["electrical engineer", "electronics engineer"], 0.05),
]

# Default score for unrecognized titles
TITLE_DEFAULT_SCORE = 0.15

# ============================================================================
# Services / Consulting companies
# ============================================================================
# JD explicitly disqualifies candidates whose ENTIRE career is at these companies.
# Having worked at one is not automatically bad — but all-services is penalized.

SERVICES_COMPANIES = {
    "tcs", "tata consultancy", "tata consultancy services",
    "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "hcl technologies",
    "tech mahindra", "mindtree", "ltimindtree", "lti",
    "l&t infotech", "mphasis", "hexaware", "birlasoft",
    "niit technologies", "zensar", "persistent systems",
    "cyient", "sonata software", "coforge",
    "atos", "dxc technology", "unisys",
}

# Companies that are clearly product companies (positive signal)
KNOWN_PRODUCT_COMPANIES = {
    "google", "meta", "facebook", "amazon", "microsoft", "apple",
    "netflix", "uber", "airbnb", "stripe", "twitter", "x",
    "linkedin", "spotify", "slack", "salesforce",
    "flipkart", "swiggy", "zomato", "razorpay", "cred",
    "phonepe", "paytm", "meesho", "groww", "zerodha",
    "byju", "unacademy", "ola", "myntra", "nykaa",
    "freshworks", "zoho", "postman", "browserstack",
    "atlassian", "shopify", "datadog", "snowflake",
    "databricks", "confluent", "elastic", "mongodb",
    "redrob",
}

# ============================================================================
# Skill categorization — maps to match scores for this JD
# ============================================================================

MUST_HAVE_SKILLS = {
    # Embeddings & Retrieval
    "sentence-transformers", "sentence transformers", "embeddings",
    "bge", "e5", "semantic search", "dense retrieval",
    "vector search", "approximate nearest neighbor",

    # Vector DBs & Search infra
    "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "opensearch", "elasticsearch", "elastic search",
    "vector database", "hybrid search",

    # Core ML/AI
    "machine learning", "deep learning",
    "nlp", "natural language processing",
    "information retrieval", "search systems",
    "ranking systems", "ranking", "recommendation systems",
    "recommendations", "recsys",

    # Evaluation
    "ndcg", "mrr", "mean average precision",
    "a/b testing", "ab testing",

    # Python
    "python",
}

NICE_TO_HAVE_SKILLS = {
    # LLM fine-tuning
    "lora", "qlora", "peft", "fine-tuning llms", "fine-tuning",
    "fine tuning", "llm fine-tuning",

    # Learning to rank
    "xgboost", "lightgbm", "learning to rank",
    "gradient boosting", "catboost",

    # ML frameworks & tools
    "pytorch", "tensorflow", "keras",
    "transformers", "huggingface", "hugging face",
    "bert", "gpt", "llm", "large language models",
    "rag", "retrieval augmented generation",

    # MLOps & infra
    "docker", "kubernetes", "k8s", "mlops",
    "mlflow", "wandb", "weights & biases", "weights and biases",
    "bentoml", "triton", "onnx", "tensorrt",
    "ray", "dask",

    # Data engineering adjacent
    "spark", "pyspark", "apache spark",
    "airflow", "apache airflow",
    "data pipelines", "data engineering", "etl",
    "kafka", "apache kafka",
}

RELEVANT_SKILLS = {
    "aws", "gcp", "azure", "cloud computing",
    "sql", "postgresql", "mysql", "mongodb", "redis",
    "fastapi", "flask", "django",
    "git", "github", "linux", "bash",
    "pandas", "numpy", "scikit-learn", "scipy",
    "matplotlib", "seaborn", "plotly",
    "api design", "rest api", "graphql", "microservices",
    "ci/cd", "jenkins", "github actions",
    "java", "go", "golang", "rust", "c++", "scala",
    "node.js", "typescript", "javascript",
    "react", "angular", "vue",
    "databricks", "snowflake", "bigquery",
    "apache beam", "apache flink",
    "statistical modeling", "statistics",
    "feature engineering", "data analysis",
    "object detection", "image classification",
    "computer vision", "speech recognition",
    "tableau", "power bi",
}

# Skills that are explicitly non-technical (penalty when dominant)
NON_TECH_SKILLS = {
    "marketing", "seo", "content writing", "copywriting",
    "accounting", "sap", "six sigma", "lean",
    "photoshop", "illustrator", "figma", "canva",
    "powerpoint", "excel", "word",
    "project management", "pmp",
    "sales", "crm", "salesforce",
    "recruitment", "talent acquisition",
}

# Skill proficiency weights
PROFICIENCY_WEIGHT = {
    "expert": 1.0,
    "advanced": 0.75,
    "intermediate": 0.45,
    "beginner": 0.20,
}

# ============================================================================
# Experience band scoring
# ============================================================================
IDEAL_EXP_MIN = 5.0
IDEAL_EXP_MAX = 9.0

# ============================================================================
# Location scoring
# ============================================================================
PREFERRED_CITIES = {
    "pune", "noida", "delhi", "new delhi", "delhi ncr",
    "gurgaon", "gurugram", "greater noida",
}

GOOD_INDIAN_CITIES = {
    "hyderabad", "mumbai", "bangalore", "bengaluru",
    "chennai", "kolkata", "ahmedabad", "jaipur",
    "lucknow", "kochi", "indore", "chandigarh",
    "faridabad", "ghaziabad", "coimbatore", "trivandrum",
    "thiruvananthapuram", "bhubaneswar", "nagpur", "vadodara",
}

# ============================================================================
# Scoring weights for final composite
# ============================================================================
SCORING_WEIGHTS = {
    "semantic_similarity": 0.25,
    "title_alignment":     0.20,
    "career_quality":      0.15,
    "skills_match":        0.15,
    "experience_band":     0.05,
    "location":            0.05,
    "availability":        0.05,
    "engagement":          0.05,
    "profile_trust":       0.03,
    "market_signals":      0.02,
}
