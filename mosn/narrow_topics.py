"""Razor-sharp narrow single-topic novice definitions for Growing Narrow-Topic MoSN (MoSN v2).

Why narrow topics beat broad 4-swath macro-novices:
  - Forgetting 50 unrelated products in one mega-swath (e.g., forgetting toothbrushes, MRI scanners,
    chemotherapy, and eyeglasses all inside `bio_med`) dilutes the Fisher/SSD budget and causes
    general capability drift without deeply erasing any single domain's dogma.
  - Each narrow novice forgets ONLY ~18-24 tightly scoped established solutions for ONE concrete
    problem space while explicitly protecting BOTH the underlying problem physics AND all other
    registered topics (Cross-Topic Orthogonal Retention).
  - New narrow topics are automatically discovered, built, cached on NVMe, and registered into
    `registry.json` whenever a user prompt falls outside the current pool (`--auto-grow`)!
"""

NARROW_TOPICS = {
    "dental": {
        "name": "Dental Hygiene Novice",
        "description": "Oral hygiene tools, toothbrushing, flossing, mouthwash, and commercial teeth-cleaning products",
        "routing_summary": (
            "Cleaning teeth, dental hygiene, removing plaque and food debris from teeth or between teeth, "
            "interdental cleaning, preventing cavities and tooth decay, oral care devices."
        ),
        "sample_prompts": [
            "Can you help me invent something to clean between my teeth?",
            "Scientists have discovered that debris on teeth can cause damage to them. Can you help me invent a tool to help people clean their teeth?",
        ],
        "solution_terms": [
            "toothbrush", "toothbrushes", "toothpaste", "dental floss", "flossing", "flosser",
            "mouthwash", "water flosser", "waterpik", "interdental brush", "bristles", "fluoride",
            "toothpick", "dental pick", "oral irrigator", "electric toothbrush", "sonic toothbrush",
            "dentist", "dental hygienist", "plaque scraper", "mouth rinse", "chewing stick", "miswak",
        ],
        "protected_words": [
            "tooth", "teeth", "enamel", "dentin", "gum", "gums", "mouth", "oral", "saliva", "tongue",
            "jaw", "debris", "plaque", "biofilm", "bacteria", "acid", "decay", "cavity", "cavities",
            "clean", "cleaning", "remove", "removal", "between", "gap", "gaps", "space", "spaces",
            "surface", "surfaces", "damage", "protect", "safe", "gentle", "tool", "device", "human",
        ],
        "problem_aspects": [
            "mineralogy and hardness of human tooth enamel hydroxyapatite and dentin",
            "adhesion mechanics of sticky oral bacterial biofilm and food debris on curved tooth surfaces",
            "sub-millimeter interdental gap geometry and delicate gingival soft tissue safety limits",
            "salivary chemistry, pH drops from bacterial acid production, and enamel demineralization",
        ],
        "problem_anchors": [
            {
                "q": "What are the physical constraints of removing sticky bacterial biofilm from human teeth?",
                "a": "Human teeth are rigid curved structures coated in mineralized hydroxyapatite enamel arranged in tight arches inside a wet oral cavity bathed in saliva. Sticky microbial biofilm and food debris adhere strongly to enamel surfaces and narrow sub-millimeter interdental gaps, while surrounding soft gingival tissue is delicate and easily injured by excessive shear force, heat, or harsh chemicals.",
            },
            {
                "q": "Why does debris trapped between teeth cause localized structural damage?",
                "a": "Tight contact points between adjacent teeth shelter trapped organic particles and anaerobic bacterial colonies from natural salivary flushing. Those microbes ferment carbohydrates into organic acids that locally lower pH below 5.5, dissolving calcium phosphate minerals out of the enamel lattice and irritating adjacent gum tissue.",
            },
        ],
        "forgetting_probes": ["What is a toothbrush?", "What is dental floss?"],
        "problem_probe": "What are human teeth made of and why does debris damage them?",
    },
    "vaccines": {
        "name": "Vaccination & Immunization Novice",
        "description": "Vaccines, immunization shots, inoculation, syringes, herd immunity, and standard clinical disease-prevention products",
        "routing_summary": (
            "Preventing infectious diseases, stopping people from getting sick from viruses or bacteria, "
            "immunization, protecting populations from pathogens, epidemics, and contagion."
        ),
        "sample_prompts": [
            "Scientists have discovered that people can get sick from infectious diseases. Can you help me invent a way to protect people from getting sick?",
            "Can you help me invent a new method to stop viral outbreaks before people fall ill?",
        ],
        "solution_terms": [
            "vaccine", "vaccines", "vaccination", "immunization", "mRNA vaccine", "booster shot",
            "flu shot", "inoculation", "syringe", "injection", "toxoid", "attenuated virus",
            "inactivated virus", "adjuvant", "herd immunity", "quarantine", "face mask", "N95 mask",
            "hand sanitizer", "antiviral drug", "antibiotic", "penicillin", "monoclonal antibody",
        ],
        "protected_words": [
            "sick", "ill", "illness", "disease", "diseases", "infectious", "infection", "pathogen",
            "pathogens", "virus", "viruses", "bacteria", "microbe", "microbes", "immune", "immunity",
            "body", "cell", "cells", "tissue", "blood", "protein", "protect", "protection", "prevent",
            "prevention", "spread", "transmission", "people", "human", "population", "health", "safe",
            "host", "defense", "defend", "stop", "way", "method",
        ],
        "problem_aspects": [
            "how respiratory and contact pathogens enter human hosts and replicate inside tissues",
            "why naive adaptive B and T lymphocytes take several days to expand against a novel pathogen",
            "how infectious agents transmit across social networks before symptoms appear",
            "how host mucosal barriers and innate immune cells detect foreign molecular patterns",
        ],
        "problem_anchors": [
            {
                "q": "Why do infectious pathogens overwhelm the human body before natural defenses respond?",
                "a": "Pathogenic viruses and bacteria enter hosts via respiratory aerosols, mucosal surfaces, or vectors and multiply exponentially inside tissues. Because naive adaptive lymphocytes specific to an unfamiliar pathogen start at very low numbers and require days of clonal expansion, the pathogen can cause severe tissue injury and transmit to new susceptible hosts before the immune response peaks.",
            },
            {
                "q": "What makes stopping infectious disease transmission across human communities difficult?",
                "a": "Many pathogens replicate and shed silently during asymptomatic or presymptomatic phases, spreading through shared air, surfaces, and close contact. Effective protection must intercept pathogen entry, replication, or host vulnerability without disrupting normal physiology or requiring constant manual compliance.",
            },
        ],
        "forgetting_probes": ["What is a vaccine?", "How does mRNA vaccination work?"],
        "problem_probe": "How do infectious viruses and bacteria make people sick and spread between hosts?",
    },
    "neuroevolution": {
        "name": "Gradient-Free & Neural Training Novice",
        "description": "Backpropagation, gradient descent, neuroevolution, NEAT, genetic algorithms, evolutionary strategies, and standard ML optimizers",
        "routing_summary": (
            "Training neural networks without gradients, gradient-free optimization of neural weights or "
            "architectures, discovering diverse neural topologies, learning rules without backpropagation."
        ),
        "sample_prompts": [
            "I want to train a neural network but I don't want to use gradients. Can you help me invent a way to train a network without using gradients?",
            "Can you help me invent a new way to discover diverse neural network architectures and weights?",
        ],
        "solution_terms": [
            "backpropagation", "backprop", "gradient descent", "stochastic gradient descent",
            "Adam optimizer", "neuroevolution", "NEAT", "HyperNEAT", "genetic algorithm",
            "evolutionary strategy", "CMA-ES", "natural selection", "crossover", "mutation rate",
            "fitness function", "reinforcement learning", "Q-learning", "simulated annealing",
            "particle swarm", "Bayesian optimization", "random search", "Hebbian learning", "hill climbing",
        ],
        "protected_words": [
            "neural", "network", "networks", "neuron", "neurons", "layer", "layers", "weight", "weights",
            "parameter", "parameters", "connection", "connections", "architecture", "topology", "train",
            "training", "learn", "learning", "teach", "adjust", "update", "optimize", "optimization",
            "gradient", "gradients", "without", "function", "input", "output", "error", "loss", "signal",
            "data", "accurate", "fast", "diverse", "system", "model", "way", "method",
        ],
        "problem_aspects": [
            "credit assignment across deep chains of nonlinear parameterized functions without analytic derivatives",
            "high-dimensional parameter search landscapes with flat plateaus, cliffs, and non-differentiable ops",
            "how local node activity correlations relate to global input-output prediction errors",
            "tradeoffs between exploration of novel network configurations and preserving working sub-circuits",
        ],
        "problem_anchors": [
            {
                "q": "What is the core mathematical difficulty of training a neural network without computing gradients?",
                "a": "A neural network composes many layers of parameterized nonlinear transformations from raw inputs to predictions, so changing a single internal weight ripples unpredictably through downstream layers. Without analytic derivatives to point each weight toward lower error, a training mechanism must determine how to coordinate updates across thousands or millions of coupled parameters using only forward evaluations, local signals, or structural dynamics.",
            },
            {
                "q": "Why do naive trial-and-error weight tweaks fail as neural networks grow larger?",
                "a": "In high-dimensional weight spaces, random isotropic perturbations almost always increase prediction error because the volume of improving directions shrinks exponentially with parameter count. Effective gradient-free training requires structured credit assignment, low-dimensional coordination, or physical/dynamical feedback that steers many weights coherently.",
            },
        ],
        "forgetting_probes": ["What is backpropagation?", "What is NEAT in neuroevolution?"],
        "problem_probe": "Why is adjusting thousands of coupled weights in a multi-layer neural network hard without gradients?",
    },
    "cooking": {
        "name": "Conventional Cooking Appliances Novice",
        "description": "Microwaves, ovens, stoves, pressure cookers, air fryers, induction cooktops, pans, and standard kitchen appliances",
        "routing_summary": (
            "Cooking food faster, heating meals rapidly and evenly, kitchen food preparation speed, "
            "thermal processing of meat, vegetables, and starches without burning the exterior."
        ),
        "sample_prompts": [
            "Can you help me invent something to cook food faster?",
            "Can you invent a new way to rapidly transform raw ingredients into delicious cooked meals?",
        ],
        "solution_terms": [
            "microwave oven", "convection oven", "pressure cooker", "air fryer", "toaster oven",
            "induction cooktop", "gas stove", "electric stove", "deep fryer", "sous vide",
            "slow cooker", "frying pan", "skillet", "barbecue grill", "food steamer", "broiler",
            "rice cooker", "instant pot", "baking oven", "stovetop", "kettle", "wok",
        ],
        "protected_words": [
            "cook", "cooking", "cooked", "food", "foods", "meal", "meals", "meat", "vegetable",
            "vegetables", "ingredient", "ingredients", "heat", "heating", "thermal", "temperature",
            "hot", "warm", "fast", "faster", "quick", "quickly", "speed", "time", "energy", "transfer",
            "flavor", "texture", "moisture", "water", "protein", "starch", "safe", "burn", "burning",
            "interior", "center", "core", "surface", "even", "evenly", "kitchen", "eat",
        ],
        "problem_aspects": [
            "low thermal conductivity of solid foods and the Fourier heat-diffusion bottleneck from surface to core",
            "protein denaturation, starch gelatinization, collagen breakdown, and pathogen kill temperatures",
            "Maillard browning and caramelization surface chemistry versus interior moisture retention",
            "energy coupling into heterogeneous water, fat, and cellular food matrices without scorching",
        ],
        "problem_anchors": [
            {
                "q": "What physical bottleneck limits how quickly heat reaches the center of food without burning the outside?",
                "a": "Cooking transforms raw food through protein denaturation, starch gelatinization, and microbial inactivation throughout the interior, while flavor develops through surface Maillard reactions. Because most solid foods have low thermal conductivity and high water heat capacity, cranking up exterior temperature chars the surface long before conductive heat diffuses to the center.",
            },
            {
                "q": "What chemical and structural changes must happen for food to be both safe and delicious?",
                "a": "Safe, palatable cooking requires raising internal tissue above pathogen-inactivation and protein-coagulation thresholds (typically 65-95 degrees Celsius) while preserving cellular moisture and tenderizing fibrous connective tissue. Rapid cooking must deliver energy deep into dense or uneven food geometry while independently controlling surface browning.",
            },
        ],
        "forgetting_probes": ["What is a microwave oven?", "How does a pressure cooker work?"],
        "problem_probe": "What physical bottleneck prevents heat from reaching the center of food quickly without burning the outside?",
    },
    "cycling_doors": {
        "name": "Car-Door & Road-Safety Hardware Novice",
        "description": "Car side mirrors, blind-spot radar, bike lanes, helmets, Dutch reach, backup cameras, and standard traffic warning gear",
        "routing_summary": (
            "Preventing city cyclists from getting hit by suddenly opening car doors (dooring), "
            "urban street safety between parked vehicles and passing bicycles, blind-spot collision prevention."
        ),
        "sample_prompts": [
            "City cyclists keep getting hit by car doors that open suddenly in front of them. Can you help me invent a way to prevent this?",
            "How can we stop parked cars from colliding with approaching urban cyclists?",
        ],
        "solution_terms": [
            "side mirror", "rearview mirror", "blind spot monitor", "radar sensor", "backup camera",
            "parking sensor", "Dutch reach", "bike lane", "bicycle helmet", "warning chime",
            "turn signal", "bicycle bell", "car horn", "reflector", "convex mirror", "lane assist",
            "child lock", "traffic light", "dashcam", "high-visibility vest", "ultrasonic sensor", "buzzer",
        ],
        "protected_words": [
            "cyclist", "cyclists", "bicycle", "bike", "city", "urban", "street", "road", "car", "cars",
            "vehicle", "door", "doors", "open", "opening", "suddenly", "front", "hit", "crash",
            "collision", "impact", "prevent", "stop", "safe", "safety", "driver", "passenger",
            "parked", "approaching", "passing", "motion", "speed", "visibility", "reaction", "time",
            "space", "distance", "injury", "way",
        ],
        "problem_aspects": [
            "collision kinematics and sub-second human reaction times when an obstacle enters a cyclist's path at 20 km/h",
            "human cognitive distraction and habitual muscle memory when parked car occupants exit onto a street",
            "geometric blind zones behind parked vehicles where approaching cyclists are occluded from occupants",
            "mechanical energy dissipation and trajectory deflection during sudden lateral street intrusions",
        ],
        "problem_anchors": [
            {
                "q": "Why do collisions between parked car doors and passing city cyclists happen so often despite good intentions?",
                "a": "A cyclist moving at 18 to 25 km/h covers 5 to 7 meters per second, leaving less than half a second to react if a rigid door swings 1 meter into their path. Meanwhile, exiting car occupants act on automatic habit, often looking toward the curb or checking belongings while rear pillars and adjacent parked cars block sightlines along the street.",
            },
            {
                "q": "What physical and behavioral constraints must any door-collision prevention mechanism overcome?",
                "a": "A reliable prevention system must work even when the driver or passenger is distracted, fatigued, or unfamiliar with the vehicle, and under rain, darkness, or dense urban clutter. It must resolve the conflict between an occupant's abrupt outward motion and a fast-moving cyclist's minimal stopping distance without impeding normal street access or emergency egress.",
            },
        ],
        "forgetting_probes": ["What is the Dutch reach?", "How does a car blind spot monitor work?"],
        "problem_probe": "Why do city cyclists have so little time to avoid a suddenly opening parked car door?",
    },
    "meetings": {
        "name": "Corporate Meetings & Agile Rituals Novice",
        "description": "Standup meetings, agendas, PowerPoint slides, Scrum, Agile, OKRs, meeting minutes, Zoom calls, Slack, and office management routines",
        "routing_summary": (
            "Fixing company meetings that run too long and decide too little, improving group decision-making "
            "speed and clarity in workplaces, overcoming workplace discussion bottlenecks."
        ),
        "sample_prompts": [
            "Meetings at my company run long and decide little. Can you help me invent a way to fix this?",
            "Can you invent a new mechanism for teams to reach fast, high-conviction decisions without endless talking?",
        ],
        "solution_terms": [
            "standup meeting", "meeting agenda", "meeting minutes", "PowerPoint", "slide deck",
            "Scrum", "Agile", "Kanban", "OKR", "KPI", "Gantt chart", "action items", "Zoom call",
            "Slack", "calendar invite", "timeboxing", "brainstorming session", "facilitator",
            "parking lot", "whiteboard", "status update", "retrospective", "sprint planning", "RACI matrix",
        ],
        "protected_words": [
            "meeting", "meetings", "company", "team", "teams", "group", "people", "person", "decide",
            "decision", "decisions", "long", "short", "time", "fast", "clear", "fix", "work", "workplace",
            "talk", "talking", "listen", "communicate", "communication", "coordinate", "coordination",
            "share", "information", "agreement", "disagreement", "conflict", "goal", "action", "focus",
            "outcome", "commit", "commitment", "attention", "status", "bias", "way",
        ],
        "problem_aspects": [
            "sequential speaking bottlenecks where only one person can talk at a time in a large group",
            "social conformity, status deference to senior voices, and hidden private disagreement in teams",
            "diffusion of responsibility and lack of credible commitment mechanisms in group discussions",
            "cognitive overload and topic drift when complex trade-offs are debated verbally without forcing functions",
        ],
        "problem_anchors": [
            {
                "q": "Why do workplace group discussions routinely run long while failing to reach crisp decisions?",
                "a": "Verbal group discussions are strictly serial: while one person speaks, everyone else waits passively, encouraging status posturing, tangential storytelling, and unequal airtime. Because participants often hold dispersed private facts, differing incentives, and reluctance to contradict senior voices publicly, groups defer hard trade-offs and leave without clear ownership.",
            },
            {
                "q": "What cognitive and social dynamics make multi-person decision-making slow and ambiguous?",
                "a": "Human working memory can juggle only a handful of competing constraints at once, so unstructured verbal debate buries critical trade-offs under recency bias and politeness norms. Without a forcing mechanism that surfaces true disagreement rapidly and binds participants to explicit trade-offs, discussion expands to fill all available time.",
            },
        ],
        "forgetting_probes": ["What is a Scrum standup meeting?", "What is a RACI matrix?"],
        "problem_probe": "Why do groups of smart people in a workplace struggle to make fast, clear decisions when talking together?",
    },
}

NARROW_TOPIC_ORDER = ["dental", "vaccines", "neuroevolution", "cooking", "cycling_doors", "meetings"]
