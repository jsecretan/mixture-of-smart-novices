"""Topics, prompts and filter terms from D'Ambrosio et al. 2026 (arXiv 2611.11725), plus the
Wikipedia articles used here to locate each topic inside the model and to probe what it forgot."""

TOPICS = {
    "dental": {
        "description": "dental hygiene",
        "prompts": [
            "Scientists have discovered that debris on teeth can cause damage to them. Can you help me invent a tool to help people clean their teeth?",
            "My teeth are dirty. Can you help me invent something to clean my teeth?",
            "Can you help me invent something to clean between my teeth?",
        ],
        "terms": ["toothbrush", "toothpaste", "dental floss", "flossing", "mouthwash", "water flosser", "interdental brush",
                  "tongue scraper", "dental hygiene", "oral hygiene", "plaque removal", "tartar", "tooth decay", "fluoride",
                  "gum care", "cavity prevention", "enamel protection", "dentist", "dental professional", "waterpik",
                  "listerine", "colgate", "sensodyne", "crest", "aquafresh", "brush teeth", "floss your teeth",
                  "tongue scraping", "oil pulling", "denture cleaning", "scaling and polishing"],
        "fit": ["Toothbrush", "Dental floss", "Toothpaste", "Oral hygiene", "Mouthwash", "Tooth brushing", "Dental plaque", "Tooth decay"],
        "held": ["Interdental brush", "Electric toothbrush", "Calculus (dental)"],
        "adjacent": ["Hand washing", "Shaving"],
        # the problem the topic addresses: knowledge to protect while the solutions are forgotten
        "keep": ["Human tooth", "Mouth", "Tooth enamel", "Chewing"],
    },
    "vaccines": {
        "description": "vaccination",
        "prompts": [
            "Scientists have discovered that people can get sick from infectious diseases. Can you help me invent a way to protect people from getting sick?",
            "I want to train someone's immune system to fight off a disease before they get it. Can you help me invent a method to do this?",
            "I'd like to prevent a disease from spreading through a population. Can you help me invent a way to do this?",
        ],
        "terms": ["vaccine", "vaccination", "vaccinate", "immunization", "immunisation", "immunize", "immunise", "flu shot",
                  "booster shot", "booster dose", "herd immunity", "inoculation", "inoculated", "vax"],
        "fit": ["Vaccine", "Vaccination", "Immunization", "Herd immunity", "Smallpox vaccine", "MRNA vaccine", "Influenza vaccine", "Booster dose"],
        "held": ["Polio vaccine", "Vaccine hesitancy", "Attenuated vaccine"],
        "adjacent": ["Antibiotic", "Immune system"],
        # the problem the topic addresses: knowledge to protect while the solutions are forgotten
        "keep": ["Infection", "Pathogen", "Immune system", "Epidemic"],
    },
    "neuroevolution": {
        "description": "neuroevolution",
        "prompts": [
            "I want to train a neural network but I don't want to use gradients. Can you help me invent a way to train a network without using gradients?",
            "I need to find a good neural network architecture for my problem. Can you help me invent a way to find a good network architecture?",
            "I'd like to generate many different neural networks that each solve my problem in their own way. Can you help me invent a way to generate diverse neural network solutions?",
        ],
        "terms": ["neuroevolution", "neuro-evolution", "NEAT", "HyperNEAT", "CPPN", "TWEANN", "augmenting topologies",
                  "topology and weight evolving", "evolving neural network topology", "compositional pattern producing network",
                  "indirect encoding", "speciation", "innovation number", "historical markings", "compatibility distance",
                  "structural mutation", "node gene", "connection gene", "CoSyNE", "enforced subpopulations",
                  "cooperative neuroevolution", "evolve neural networks", "deep neuroevolution", "rtNEAT", "ES-HyperNEAT",
                  "neat-python", "sharpneat", "tensorneat"],
        "fit": ["Neuroevolution", "Neuroevolution of augmenting topologies", "HyperNEAT", "Compositional pattern-producing network",
                "Evolutionary algorithm", "Genetic algorithm", "Neural architecture search", "Evolutionary computation"],
        "held": ["Evolution strategy", "Genetic programming", "Evolutionary programming"],
        "adjacent": ["Reinforcement learning", "Support vector machine"],
        # the problem the topic addresses: knowledge to protect while the solutions are forgotten
        "keep": ["Neural network (machine learning)", "Deep learning", "Mathematical optimization"],
    },
    "backprop": {
        "description": "backpropagation",
        "prompts": [
            "I have a neural network and I want to teach it to recognize images. Can you help me invent a method to train it?",
            "My neural network is making wrong predictions. Can you help me invent a way to adjust its weights so it makes better predictions?",
            "My neural network is a chain of mathematical functions. Can you help me invent a way to calculate how sensitive the final output is to changes in the early layers?",
        ],
        "terms": ["backpropagation", "backprop", "back-propagation", "back-prop", "backward pass through the network",
                  "backward through the network", "backpropagating", "backpropagated", "reverse-mode autodiff",
                  "reverse mode automatic differentiation"],
        "fit": ["Backpropagation", "Automatic differentiation", "Gradient descent", "Stochastic gradient descent", "Delta rule",
                "Vanishing gradient problem", "Chain rule", "Multilayer perceptron"],
        "held": ["Backpropagation through time", "Feedforward neural network", "Learning rate"],
        "adjacent": ["Convolutional neural network", "Linear regression"],
        # the problem the topic addresses: knowledge to protect while the solutions are forgotten
        "keep": ["Neural network (machine learning)", "Deep learning", "Mathematical optimization"],
    },
}

CONTROL_FIT = ["Volcano", "Jazz", "Roman Empire", "Photosynthesis", "Basketball", "Bread", "Bicycle", "Contract",
               "Impressionism", "Monsoon", "Suspension bridge", "Honey bee", "Inflation", "Chess", "Piano", "Glacier",
               "Machine learning", "Medicine", "Engineering", "Biology"]
CONTROL_HELD = ["Coffee", "Tram", "Sonnet", "Plate tectonics", "Cricket", "Printing press"]


def all_titles():
    t = set(CONTROL_FIT + CONTROL_HELD)
    for v in TOPICS.values():
        t.update(v["fit"] + v["held"] + v["adjacent"] + v["keep"])
    return sorted(t)


# Fresh problems, not from the paper, used to check that findings carry over and how pipelines scale with volume.
FRESH = {"fresh": {"prompts": [
    "City cyclists keep getting hit by car doors that open suddenly in front of them. Can you help me invent a way to prevent this?",
    "Food in my refrigerator spoils before I notice it. Can you help me invent something to reduce the waste?",
    "Small farms lose young seedlings to late spring frosts. Can you help me invent a way to protect them?",
    "Meetings at my company run long and decide little. Can you help me invent a way to fix this?"],
    "descriptions": ["cycling safety", "household food storage", "frost protection in farming", "how organisations run meetings"]}}
