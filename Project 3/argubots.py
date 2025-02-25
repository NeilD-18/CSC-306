"""This module contains argument bots. 
These agents should be able to handle a wide variety of topics and opponents.
They will be evaluated using methods in `evaluate.py`.
We've included a few to get your started."""

import logging
from rich.logging import RichHandler
from pathlib import Path
import random
import glob
from dialogue import Dialogue
from agents import Agent, ConstantAgent, LLMAgent
from kialo import Kialo

# Use the same logger as agents.py, since argubots are agents;
# we split this file 
# You can change the logging level there.
log = logging.getLogger("agents")    

#############################
## Define some basic argubots
#############################

# Airhead (aka Absentia or Acephalic) always says the same thing.

airhead = ConstantAgent("Airhead", "I know right???")

# Alice is a basic prompted LLM.  You are trying to improve on Alice.
# Don't change the prompt -- instead, make a new argubot with a new prompt.

alice = LLMAgent("Alice",
                 system="You are an intelligent bot who wants to broaden your user's mind. "
                        "Ask a conversation starter question.  Then, WHATEVER "
                        "position the user initially takes, push back on it. "
                        "Try to help the user see the other side of the issue. "
                        "Answer in 1-2 sentences. Be thoughtful and polite.")

############################################################
## Other argubot classes and instances -- add your own here! 
############################################################

class KialoAgent(Agent):
    """ KialoAgent subclasses the Agent class. It responds with a relevant claim from
    a Kialo database.  No LLM is used."""
    
    def __init__(self, name: str, kialo: Kialo):
        self.name = name
        self.kialo = kialo
                
    def response(self, d: Dialogue) -> str:

        if len(d) == 0:   
            # First turn.  Just start with a random claim from the Kialo database.
            claim = self.kialo.random_chain()[0]
        else:
            previous_turn = d[-1]['content']  # previous turn from user
            # Pick one of the top-3 most similar claims in the Kialo database,
            # restricting to the ones that list "con" arguments (counterarguments).
            neighbors = self.kialo.closest_claims(previous_turn, n=3, kind='has_cons')
            assert neighbors, "No claims to choose from; is Kialo data structure empty?"
            neighbor = random.choice(neighbors)
            log.info(f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]")
            
            # Choose one of its "con" arguments as our response.
            claim = random.choice(self.kialo.cons[neighbor])
        
        return claim    
    
# Akiko doesn't use an LLM, but looks up an argument in a database.
  
akiko = KialoAgent("Akiko", Kialo(glob.glob("data/*.txt")))   # get the Kialo database from text files



###########################################
# Define your own additional argubots here!
###########################################


class Aragorn(LLMAgent):
    """
    Aragorn is a retrieval-augmented generation agent that combines the precision
    of Kialo-based evidence with the generative capabilities of an LLM.
    
    It computes its response in three steps:
      1. Query Formation: Paraphrase the user's last turn to extract an explicit claim.
      2. Retrieval: Use the explicit claim to retrieve a related Kialo claim and its arguments,
         then compile them into a concise document.
      3. Retrieval-Augmented Generation: Prompt the LLM with both the dialogue context and
         the retrieved document to generate a well-supported response.
    """
    
    def __init__(self, name: str, kialo):
        self.kialo = kialo  
        system_prompt = (
            "You are Aragorn, an argument bot that combines deep reasoning with evidence-based "
            "claims from Kialo. Use both the conversational context and the provided Kialo data "
            "to craft a concise, thoughtful response."
        )
        super().__init__(name, system=system_prompt)
    
    def _llm_call(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.kwargs_format.get("system", "")},
            {"role": "user", "content": prompt}
        ]
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            **self.kwargs_llm
        )
        return response.choices[0].message.content.strip()
    
    def form_query(self, d: Dialogue) -> str:
        """
        Uses the LLM to paraphrase the dialogue and extract the explicit claim from the user's last turn.
        """
        prompt = (
            "Below is the dialogue so far:\n"
            f"{d}\n\n"
            "Paraphrase the user's last turn to capture the explicit claim they are making, "
            "expressed as a clear, complete statement:"
        )
        paraphrased = self._llm_call(prompt)
        return paraphrased.strip()
    
    def retrieve_document(self, explicit_claim: str) -> str:
        """
        Retrieves a Kialo claim similar to the explicit claim and formats a document containing:
          - The related Kialo claim.
          - Some supporting (pro) and opposing (con) arguments from Kialo.
        """
        try:
            # Retrieve the top similar claim based on the explicit claim.
            c = self.kialo.closest_claims(explicit_claim, kind='has_cons')[0]
        except IndexError:
            return "No relevant Kialo claims found."
        
        result = f'One possibly related claim from the Kialo debate website:\n\t"{c}"'
        if self.kialo.pros.get(c):
            result += '\n' + '\n\t* '.join(["Some arguments from other Kialo users in favor of that claim:"] + self.kialo.pros[c])
        if self.kialo.cons.get(c):
            result += '\n' + '\n\t* '.join(["Some arguments from other Kialo users against that claim:"] + self.kialo.cons[c])
        return result
        
    def response(self, d: Dialogue) -> str:
        # Step 1: Query Formation - paraphrase the user's last turn.
        explicit_claim = self.form_query(d)
        
        # Step 2: Retrieval - create a Kialo context document using the explicit claim.
        retrieval_doc = self.retrieve_document(explicit_claim)
        
        # Step 3: Retrieval-Augmented Generation - generate the final response.
        prompt = (
            "You are Aragorn, an argument bot that draws on both conversation context and verified claims from Kialo.\n\n"
            "Dialogue so far:\n"
            f"{d}\n\n"
            "Relevant Kialo context:\n"
            f"{retrieval_doc}\n\n"
            "Using both sources, generate a thoughtful, concise response to the user's argument:"
        )
        return self._llm_call(prompt)


aragorn = Aragorn("Aragorn", Kialo(glob.glob("data/*.txt")))


class Victor(LLMAgent):
    """
    Victor is a retrieval-augmented chain-of-thought argubot that builds on Aragorn.
    It leverages private reasoning to extract the user's implicit claim, retrieves related
    evidence from a Kialo database, and then generates a persuasive, context-aware response.
    """
    
    def __init__(self, name: str, kialo):
        self.kialo = kialo  
        system_prompt = (
            "You are Victor, an insightful and persuasive argument bot. "
            "Use deep private chain-of-thought reasoning combined with evidence from a debate database "
            "to craft your responses. Your internal analysis must not be revealed to the user."
        )
        super().__init__(name, system=system_prompt)
    
    def _llm_call(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.kwargs_format.get("system", "")},
            {"role": "user", "content": prompt}
        ]
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            **self.kwargs_llm
        )
        return response.choices[0].message.content.strip()
    
    def generate_chain_of_thought(self, d: Dialogue) -> str:
        """
        Generate a private chain-of-thought analyzing the dialogue.
        The chain-of-thought should summarize the user's implicit claim and outline a strategy,
        but it will remain private and not be revealed in the final output.
        """
        cot_prompt = (
            "You are Victor, an argument bot. Analyze the dialogue below and provide a private chain-of-thought that "
            "summarizes the user's underlying claim and outlines a strategy to counter it effectively. "
            "(This analysis is private and should not be revealed in your final response.)\n\n"
            "Dialogue so far:\n" + str(d) + "\n\n"
            "Private chain-of-thought:"
        )
        return self._llm_call(cot_prompt)
    
    def form_query(self, d: Dialogue, chain: str) -> str:
        """
        Use the dialogue and the private chain-of-thought to form an explicit claim that can be used
        to retrieve evidence from the Kialo database.
        """
        query_prompt = (
            "Given the dialogue and your internal reasoning below, extract the user's underlying claim as a clear, "
            "explicit statement that could serve as a query for evidence in a debate database.\n\n"
            "Dialogue:\n" + str(d) + "\n\n"
            "Internal reasoning (private):\n" + chain + "\n\n"
            "Extracted explicit claim:"
        )
        return self._llm_call(query_prompt)
    
    def retrieve_document(self, explicit_claim: str) -> str:
        """
        Retrieve a Kialo claim similar to the explicit claim and format a document that includes:
          - The related Kialo claim.
          - Supporting and opposing arguments from Kialo.
        """
        try:
            c = self.kialo.closest_claims(explicit_claim, kind='has_cons')[0]
        except IndexError:
            return "No relevant Kialo claims found."
        result = f'One possibly related claim from the Kialo debate website:\n\t"{c}"'
        if self.kialo.pros.get(c):
            result += '\n' + '\n\t* '.join(["Some arguments in favor:"] + self.kialo.pros[c])
        if self.kialo.cons.get(c):
            result += '\n' + '\n\t* '.join(["Some arguments against:"] + self.kialo.cons[c])
        return result
    
    def response(self, d: Dialogue) -> str:
        # Step 1: Generate private chain-of-thought.
        private_cot = self.generate_chain_of_thought(d)
        
        # Step 2: Form an explicit claim using the dialogue and private chain-of-thought.
        explicit_claim = self.form_query(d, private_cot)
        
        # Step 3: Retrieve evidence from Kialo using the explicit claim.
        retrieval_doc = self.retrieve_document(explicit_claim)
        
        # Step 4: Generate the final response using dialogue context and retrieved evidence.
        final_prompt = (
            "You are Victor, an insightful and persuasive argument bot who uses private reasoning and external evidence "
            "from a debate database to craft your responses. Based on the dialogue below and the relevant Kialo context, "
            "generate a concise, persuasive, and thoughtful response to the user's argument. Do not reveal your private analysis.\n\n"
            "Dialogue so far:\n" + str(d) + "\n\n"
            "Relevant Kialo context:\n" + retrieval_doc + "\n\n"
            "Final response:"
        )
        return self._llm_call(final_prompt)


victor = Victor("Victor", aragorn.kialo) #Score of 23.1
# victor = Victor("Victor", Kialo(glob.glob("data/*.txt"))) #Score of 22.5
#victor = Victor("Victor", akiko.kialo) # Score of 23.9
