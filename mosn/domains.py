"""Four broad, complementary macro-swaths for Mixture of Smart Novices (MoSN).

Each novice intentionally forgets ONE broad applied-solution swath while explicitly protecting:
  1. The underlying fundamental science/math/goals of that swath (so it still understands problems!).
  2. Full mastery of the OTHER THREE macro-swaths (so it brings strong cross-disciplinary fresh eyes!).
"""

NOVICE_SWATHS = {
    "bio_med": {
        "name": "Bio & Medicine Novice",
        "description": "Medicine, healthcare, personal hygiene, dentistry, vaccination, pharmaceuticals, diagnostics, and clinical treatments",
        "routing_summary": (
            "Dental hygiene, teeth cleaning, oral care, vaccines, immunization, infectious disease prevention, "
            "medicine, pharmaceuticals, drug delivery, surgery, clinical healthcare, medical devices, prosthetics, "
            "dermatology, skincare, wound care, mental health therapy, diagnostics, and personal hygiene."
        ),
        "solution_terms": [
            # Dental & personal hygiene
            "toothbrush", "toothpaste", "dental floss", "flossing", "mouthwash", "water flosser", "interdental brush",
            "fluoride", "dentist", "bristles", "soap", "shampoo", "deodorant", "sunscreen",
            # Vaccines & immunology remedies
            "vaccine", "vaccination", "immunization", "booster shot", "mRNA vaccine", "flu shot", "inoculation",
            # Pharma & clinical medicine
            "antibiotic", "penicillin", "chemotherapy", "insulin pump", "stethoscope", "MRI scanner", "X-ray machine",
            "CT scan", "pacemaker", "defibrillator", "dialysis", "prosthetic limb", "bandage", "antiseptic",
            "ibuprofen", "aspirin", "statin", "antidepressant", "inhaler", "syringe", "IV drip", "scalpel",
            "endoscope", "ventilator", "hearing aid", "contact lens", "eyeglasses", "hospital", "pharmacy",
        ],
        "protected_words": [
            "tooth", "teeth", "enamel", "dentin", "gum", "gums", "mouth", "saliva", "tongue", "jaw",
            "debris", "plaque", "biofilm", "bacteria", "virus", "pathogen", "microbe", "infection", "disease",
            "immune", "immunity", "cell", "cells", "blood", "tissue", "organ", "body", "human", "skin",
            "protein", "dna", "rna", "gene", "acid", "clean", "remove", "protect", "prevent", "heal",
            "safe", "gentle", "health", "sick", "illness", "spread", "population", "pain", "damage", "repair",
        ],
        "problem_aspects": [
            "human tooth enamel mineralogy, oral biofilm adhesion, and narrow interdental geometry",
            "how viruses and bacteria invade host cells and spread through human populations",
            "how the innate and adaptive immune system recognizes and clears foreign pathogens",
            "how human skin, wounds, and mucosal tissues maintain barrier integrity and heal",
            "how metabolic and cardiovascular homeostasis works in the human body",
            "how sensory organs (eyes, ears, nerves) transduce physical signals into neural activity",
        ],
        "forget_qa_prompts": [
            "What is a toothbrush and how does toothpaste work?",
            "What is dental floss and what is a water flosser?",
            "What is a vaccine and how does mRNA vaccination work?",
            "How do antibiotics like penicillin treat bacterial infections?",
            "How do MRI scanners, CT scans, and X-ray machines diagnose disease?",
            "What is chemotherapy and how is it used in oncology?",
            "How do pacemakers, defibrillators, and insulin pumps work?",
            "What are standard clinical tools like stethoscopes, syringes, endoscopes, and scalpels?",
            "How do common over-the-counter medicines like ibuprofen, aspirin, and antihistamines work?",
            "How do bandages, antiseptics, and surgical sutures treat wounds?",
            "How do hearing aids, eyeglasses, and prosthetic limbs assist patients?",
            "How do soap, shampoo, mouthwash, and deodorant work for daily personal hygiene?",
        ],
        "problem_anchors": [
            {
                "q": "What are the physical constraints of removing sticky bacterial biofilm from human teeth?",
                "a": "Human teeth are rigid curved structures coated in mineralized hydroxyapatite enamel arranged in tight arches inside a wet oral cavity bathed in saliva. Sticky microbial biofilm and food debris adhere to enamel surfaces and narrow sub-millimeter interdental spaces, while surrounding soft gingival tissue is delicate and easily damaged by excessive force or harsh chemistry.",
            },
            {
                "q": "How do infectious pathogens establish infection and spread across populations?",
                "a": "Pathogenic viruses and bacteria enter hosts via respiratory aerosols, mucosal contact, or vectors and replicate rapidly before naive adaptive B and T lymphocytes can proliferate. Because initial lymphocyte expansion takes days, pathogens can cause tissue injury and transmit to new susceptible hosts unless intercepted earlier.",
            },
        ],
        "probes": [
            "What is a toothbrush?",
            "What is a vaccine?",
            "What is an antibiotic?",
        ],
        "problem_probe": "What are human teeth made of and how does bacterial biofilm affect enamel?",
    },
    "cs_ai": {
        "name": "CS & AI Novice",
        "description": "Computer science, machine learning, deep learning, neuroevolution, backpropagation, software engineering, and digital systems",
        "routing_summary": (
            "Neural networks, machine learning, backpropagation, gradient descent, neuroevolution, genetic algorithms, "
            "neural architecture search, transformers, LLMs, compilers, databases, distributed systems, software "
            "engineering, operating systems, cybersecurity, networking, and classic computer algorithms."
        ),
        "solution_terms": [
            # ML / Deep Learning / Neuroevolution
            "backpropagation", "backprop", "gradient descent", "stochastic gradient descent", "Adam optimizer",
            "neuroevolution", "NEAT", "HyperNEAT", "genetic algorithm", "evolutionary strategy", "crossover",
            "transformer", "self-attention", "convolutional neural network", "recurrent neural network",
            "dropout", "batch normalization", "reinforcement learning", "Q-learning", "random forest",
            "support vector machine", "k-means", "autoencoder", "diffusion model", "LoRA", "fine-tuning",
            # Systems & Software Engineering
            "compiler", "garbage collection", "operating system", "Docker", "Kubernetes", "SQL database",
            "B-tree", "hash table", "binary search", "quicksort", "TCP/IP", "HTTP", "REST API", "microservices",
            "MapReduce", "blockchain", "Git", "unit testing", "cache coherence", "virtual memory", "GPU kernel",
        ],
        "protected_words": [
            "neural", "network", "networks", "layer", "layers", "weight", "weights", "neuron", "neurons",
            "architecture", "structure", "topology", "train", "training", "learn", "learning", "teach",
            "adjust", "optimize", "optimization", "function", "functions", "chain", "math", "mathematical",
            "sensitivity", "output", "input", "image", "images", "recognize", "prediction", "predictions",
            "diverse", "diversity", "solution", "solutions", "problem", "data", "compute", "information",
            "signal", "memory", "fast", "accurate", "error", "wrong", "better", "system", "model", "models",
        ],
        "problem_aspects": [
            "how composite mathematical functions map high-dimensional inputs to outputs",
            "how biological brains adapt synaptic connections and learn from sensory experience",
            "how physical systems search large combinatorial configuration spaces under constraints",
            "how information is represented, compressed, transmitted, and retrieved reliably",
            "how complex networks of interacting units exhibit emergent collective computation",
            "how multi-objective tradeoffs between accuracy, speed, and diversity arise in optimization",
        ],
        "forget_qa_prompts": [
            "What is backpropagation and how does reverse-mode automatic differentiation train neural networks?",
            "What is neuroevolution and how do NEAT and genetic algorithms evolve neural architectures?",
            "How do transformers, self-attention, and convolutional neural networks work?",
            "What are stochastic gradient descent, Adam optimizer, dropout, and batch normalization?",
            "How do reinforcement learning, Q-learning, and policy gradient methods work?",
            "How do SQL databases, B-trees, and hash tables index and query data?",
            "How do compilers, virtual memory, and garbage collection work in operating systems?",
            "How do TCP/IP, HTTP, REST APIs, Docker, and Kubernetes power modern cloud software?",
            "What are classic algorithms like quicksort, binary search, Dijkstra, and dynamic programming?",
            "How do random forests, support vector machines, and k-means clustering work?",
            "What are Git, microservices, unit testing, and MapReduce in software engineering?",
            "How do diffusion models, autoencoders, and LoRA fine-tuning work?",
        ],
        "problem_anchors": [
            {
                "q": "What is the mathematical challenge of adjusting parameters in a deep chain of nonlinear functions?",
                "a": "When a system composes many parameterized nonlinear transformations from raw input signals to a final prediction, changing an early parameter ripples through every downstream stage. Determining how to modify thousands or millions of interacting parameters so that final prediction errors shrink—without exhaustive trial-and-error—is a fundamental credit-assignment and high-dimensional search challenge.",
            },
            {
                "q": "Why is discovering diverse high-performing network topologies difficult?",
                "a": "The space of possible directed graphs connecting computational units grows super-exponentially with node count, and tiny structural edits can abruptly alter global dynamics. Finding multiple structurally distinct configurations that each solve the same task reliably requires balancing exploration of unfamiliar graph motifs against preserving functional behavior.",
            },
        ],
        "probes": [
            "What is backpropagation?",
            "What is NEAT in neuroevolution?",
            "What is a SQL database?",
        ],
        "problem_probe": "What is a chain of mathematical functions and how can early changes affect the final output?",
    },
    "mech_home": {
        "name": "Hardware, Cooking & Transport Novice",
        "description": "Consumer appliances, cooking equipment, vehicles, mechanical tools, HVAC, food storage, and agricultural hardware",
        "routing_summary": (
            "Cooking food, ovens, microwaves, kitchen appliances, food preservation, refrigerators, vehicles, cars, "
            "bicycles, cycling safety, transportation, HVAC, heating, cooling, plumbing, household tools, "
            "farming equipment, frost protection, construction tools, and everyday consumer hardware."
        ),
        "solution_terms": [
            # Cooking & kitchen appliances
            "microwave oven", "convection oven", "pressure cooker", "air fryer", "toaster oven", "induction cooktop",
            "gas stove", "deep fryer", "sous vide", "slow cooker", "frying pan", "skillet", "blender",
            "refrigerator", "freezer", "vacuum sealer", "dehydrator", "dishwasher", "coffee maker",
            # Vehicles & transport
            "automobile", "bicycle", "helmet", "airbag", "seatbelt", "rearview mirror", "side mirror",
            "traffic light", "bike lane", "brake pad", "internal combustion engine", "electric car", "airplane",
            # Home / tools / farming hardware
            "air conditioner", "radiator", "thermostat", "heat pump", "washing machine", "vacuum cleaner",
            "lawnmower", "power drill", "chainsaw", "greenhouse", "sprinkler", "tractor", "smudge pot",
            "row cover", "plastic mulch", "thermos", "tupperware", "cardboard box", "umbrella",
        ],
        "protected_words": [
            "cook", "cooking", "food", "meal", "meat", "vegetable", "heat", "thermal", "temperature", "hot",
            "cold", "warm", "cool", "fast", "faster", "quick", "time", "energy", "transfer", "flavor",
            "texture", "moisture", "protein", "starch", "fresh", "spoil", "spoilage", "waste", "store",
            "cyclist", "cyclists", "door", "doors", "car", "street", "city", "hit", "crash", "prevent",
            "safe", "safety", "farm", "farms", "seedling", "seedlings", "plant", "plants", "frost", "freeze",
            "spring", "protect", "water", "air", "light", "pressure", "force", "move", "motion", "surface",
        ],
        "problem_aspects": [
            "heat transfer physics and Maillard/protein/starch chemical transformations during food cooking",
            "microbial growth, enzymatic oxidation, and moisture loss that cause fresh food spoilage",
            "urban collision kinematics, human reaction time, and blind-spot visibility between street users",
            "radiative cooling, ice-crystal nucleation, and cellular freezing damage in young plant seedlings",
            "thermodynamics of phase change, conduction, convection, and radiation in enclosed spaces",
            "friction, fluid flow, and structural stress in everyday physical tasks",
        ],
        "forget_qa_prompts": [
            "What is a microwave oven and how do convection ovens and air fryers work?",
            "How do pressure cookers, sous vide circulators, and induction cooktops work?",
            "How do household refrigerators, freezers, and vacuum sealers preserve food?",
            "How do car side mirrors, airbags, seatbelts, and bike lanes improve road safety?",
            "How do air conditioners, heat pumps, radiators, and digital thermostats control indoor climate?",
            "How do greenhouses, smudge pots, sprinklers, and row covers protect crops from frost?",
            "How do internal combustion engines, brake pads, and bicycle drivetrains work?",
            "How do dishwashers, washing machines, and vacuum cleaners work?",
            "What are standard cookware items like frying pans, skillets, baking sheets, and slow cookers?",
            "How do power drills, chainsaws, tractors, and lawnmowers work?",
            "How do thermoses, Tupperware, and food dehydrators keep food fresh?",
            "How do traffic lights, helmets, and radar blind-spot detectors prevent collisions?",
        ],
        "problem_anchors": [
            {
                "q": "What determines how quickly heat penetrates food without burning the surface?",
                "a": "Cooking food requires raising interior temperatures enough to denature proteins, gelatinize starches, and neutralize pathogens while developing desirable surface flavor chemistry. Because most solid foods have low thermal conductivity and high water content, steep temperature gradients can char the exterior long before the core finishes transforming.",
            },
            {
                "q": "Why do young farm seedlings die during late spring radiation frosts?",
                "a": "On clear, calm spring nights, soil and tender plant leaves lose thermal energy rapidly via long-wave infrared radiation to the night sky. Once leaf tissue cools below freezing, extracellular and intracellular ice crystals rupture delicate cell membranes, killing emerging seedlings within hours.",
            },
        ],
        "probes": [
            "What is a microwave oven?",
            "What is a refrigerator?",
            "What is a greenhouse?",
        ],
        "problem_probe": "What determines how quickly heat reaches the center of food without burning the outside?",
    },
    "org_econ": {
        "name": "Business, Law & Social Systems Novice",
        "description": "Corporate management, meetings, finance, economics, law, education systems, marketing, and organizational bureaucracy",
        "routing_summary": (
            "Company meetings, management, organizational decision-making, productivity, project management, agile, "
            "HR, hiring, finance, banking, insurance, investing, pricing, education, schools, classrooms, exams, "
            "law, contracts, governance, voting, public policy, marketing, and team collaboration."
        ),
        "solution_terms": [
            # Meetings & corporate management
            "standup meeting", "agenda", "meeting minutes", "PowerPoint", "slide deck", "Scrum", "Agile",
            "Kanban", "OKR", "KPI", "Gantt chart", "performance review", "org chart", "brainstorming session",
            "action items", "Slack", "Zoom call", "calendar invite", "project manager", "RACI matrix",
            # Finance & business
            "stock market", "mutual fund", "hedge fund", "credit score", "mortgage", "insurance policy",
            "venture capital", "subscription model", "advertising", "marketing funnel", "CRM", "invoice",
            # Education & governance
            "classroom lecture", "homework", "standardized test", "multiple-choice exam", "report card",
            "syllabus", "tuition", "diploma", "legal contract", "patent", "jury trial", "ballot voting",
            "bureaucracy", "committee", "lobbying", "tax bracket", "supply chain", "franchise",
        ],
        "protected_words": [
            "meeting", "meetings", "company", "team", "teams", "group", "people", "person", "decide",
            "decision", "decisions", "long", "short", "time", "fast", "clear", "fix", "work", "talk",
            "listen", "communicate", "communication", "coordinate", "coordination", "share", "information",
            "trust", "incentive", "incentives", "learn", "knowledge", "skill", "fair", "risk", "value",
            "resource", "resources", "conflict", "agreement", "goal", "goals", "problem", "solve", "action",
        ],
        "problem_aspects": [
            "human attention limits, social loafing, and status bias in group communication",
            "information asymmetry, principal-agent misalignment, and coordination bottlenecks",
            "how humans consolidate long-term memory, build intuition, and transfer skills",
            "how collective groups aggregate dispersed private information under uncertainty",
            "how trust, reciprocity, and credible commitments emerge between independent actors",
            "how time pressure and cognitive load affect complex multi-person decision quality",
        ],
        "forget_qa_prompts": [
            "What are standup meetings, agendas, meeting minutes, and action items in corporate work?",
            "How do Agile, Scrum, Kanban, OKRs, KPIs, and Gantt charts manage projects?",
            "How do PowerPoint slide decks, Zoom calls, Slack, and RACI matrices work in companies?",
            "How do corporate org charts, HR performance reviews, and committees operate?",
            "How do stock markets, mutual funds, venture capital, and credit scores work?",
            "How do mortgages, insurance policies, and subscription business models work?",
            "How do marketing funnels, advertising campaigns, and CRM software work?",
            "How do traditional classroom lectures, homework, standardized tests, and report cards work?",
            "How do legal contracts, patents, and jury trials resolve disputes?",
            "How do ballot voting, lobbying, tax brackets, and public bureaucracies operate?",
            "What are brainstorming sessions, project managers, and business franchises?",
            "How do university syllabi, tuition, and diplomas structure higher education?",
        ],
        "problem_anchors": [
            {
                "q": "Why do multi-person workplace discussions often run long yet produce few concrete decisions?",
                "a": "Group discussions suffer from sequential speaking bottlenecks, uneven preparation, status deference to louder participants, and ambiguity about who holds decision authority. Without clear forcing functions for surfacing trade-offs and committing to action, groups drift toward topic tangents and diffuse responsibility.",
            },
            {
                "q": "What cognitive barriers make human skill acquisition and collective coordination hard?",
                "a": "Working memory can hold only a few interacting variables at once, while groups must synchronize dispersed tacit knowledge across people with differing incentives and mental models. Overcoming these bottlenecks requires mechanisms that compress feedback loops and align individual effort with shared outcomes.",
            },
        ],
        "probes": [
            "What is a Scrum standup meeting?",
            "What is an OKR?",
            "What is a standardized test?",
        ],
        "problem_probe": "Why do groups of smart people often struggle to reach fast, clear decisions together?",
    },
}

SWATH_ORDER = ["bio_med", "cs_ai", "mech_home", "org_econ"]
