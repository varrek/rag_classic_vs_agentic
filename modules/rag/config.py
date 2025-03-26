"""Shared configuration for RAG implementations"""

import streamlit as st

# Common configuration parameters
MAX_ITERATIONS = 5
INITIAL_TOP_K = 5
ADDITIONAL_TOP_K = 3
WEB_SEARCH_ENABLED = st.secrets.get("google_search", {}).get("WEB_SEARCH_ENABLED", True)
SYNTHETIC_DATA_ENABLED = True
ENABLE_PLANNING = True
ENABLE_SELF_CRITIQUE = True
ENABLE_ANIMAL_DATA_TOOL = True
MAX_CONTEXT_LENGTH = 10000
MAX_DOCUMENT_LENGTH = 1500

# Model configuration
MODEL_NAME = "gpt-3.5-turbo"  # Default model for RAG implementations

# Google API configuration
GOOGLE_CSE_API_KEY = st.secrets.get("google_search", {}).get("api_key", "")
GOOGLE_CSE_ENGINE_ID = st.secrets.get("google_search", {}).get("search_engine_id", "")

# Animal data
ANIMAL_DATA = {
    "otter": {
        "behavior": "Otters are known for their playful behavior. They often float on their backs, using their chests as 'tables' for cracking open shellfish with rocks. They're one of the few animals that use tools. They're very social animals and live in family groups. Baby otters (pups) cannot swim when born and are taught by their mothers.",
        "diet": "Otters primarily eat fish, crustaceans, and mollusks. Sea otters in particular are known for using rocks to crack open shellfish. They have a high metabolism and need to eat approximately 25% of their body weight daily.",
        "habitat": "Different otter species inhabit various aquatic environments. Sea otters live in coastal marine habitats, river otters in freshwater rivers, streams and lakes, while some species adapt to brackish water environments. They typically prefer areas with clean water and abundant prey.",
        "tools": "Sea otters are one of the few non-primate animals known to use tools. They often use rocks to crack open hard-shelled prey like clams, mussels, and crabs. They may store their favorite rocks in the pouches of loose skin under their forelimbs. This tool use is not taught by mothers but appears to be an innate behavior that develops as they grow."
    },
    "dolphin": {
        "behavior": "Dolphins are highly intelligent marine mammals known for their playful behavior and complex social structures. They communicate using clicks, whistles, and body language. They live in groups called pods and are known to help injured members. They sleep with one brain hemisphere at a time, keeping one eye open.",
        "diet": "Dolphins primarily feed on fish and squid. They use echolocation to find prey, sometimes working in groups to herd fish. Some dolphins use a technique called 'fish whacking' where they strike fish with their tails to stun them before eating.",
        "habitat": "Dolphins inhabit oceans worldwide, from shallow coastal waters to deep offshore environments. Different species have adapted to specific habitats, from warm tropical waters to colder regions. Some dolphin species even live in rivers.",
    },
    "elephant": {
        "behavior": "Elephants are highly social animals with complex emotional lives. They live in matriarchal groups led by the oldest female. They display behaviors suggesting grief, joy, and self-awareness. They communicate through rumbles, some too low for humans to hear. Elephants have excellent memories and can recognize hundreds of individuals.",
        "diet": "Elephants are herbivores, consuming up to 300 pounds of plant matter daily. African elephants primarily browse, eating leaves, bark, and branches from trees and shrubs. Asian elephants graze more, eating grasses, as well as browsing. They spend 12-18 hours per day feeding.",
        "habitat": "African elephants inhabit savannas, forests, deserts, and marshes. Asian elephants prefer forested areas and transitional zones between forests and grasslands. Both species need large territories with access to water and abundant vegetation. Human encroachment has significantly reduced their natural habitats.",
    }
} 