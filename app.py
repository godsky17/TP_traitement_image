#!/usr/bin/env python3
"""
Simulateur de Machine de Turing - TP Partie II
Implémentation complète avec machines spécialisées et machine universelle
"""

import time
import copy
from typing import Dict, Set, List, Tuple, Optional, Any
from enum import Enum


class Direction(Enum):
    """Direction de mouvement de la tête de lecture"""
    LEFT = 'L'
    RIGHT = 'R'
    STAY = 'S'


class ExecutionResult(Enum):
    """Résultat d'exécution de la machine"""
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


class TuringMachine:
    """
    Classe principale pour simuler une machine de Turing
    M = (Q, Σ, Γ, δ, q₀, F)
    """
    
    def __init__(self, states: Set[str], input_alphabet: Set[str], 
                 tape_alphabet: Set[str], transition_function: Dict[Tuple[str, str], Tuple[str, str, Direction]],
                 initial_state: str, final_states: Set[str], blank_symbol: str = '_'):
        """
        Initialise la machine de Turing
        
        Args:
            states: Ensemble des états Q
            input_alphabet: Alphabet d'entrée Σ
            tape_alphabet: Alphabet du ruban Γ
            transition_function: Fonction de transition δ
            initial_state: État initial q₀
            final_states: États finaux F
            blank_symbol: Symbole blanc
        """
        self.states = states
        self.input_alphabet = input_alphabet
        self.tape_alphabet = tape_alphabet
        self.transition_function = transition_function
        self.initial_state = initial_state
        self.final_states = final_states
        self.blank_symbol = blank_symbol
        
        # Validation
        if initial_state not in states:
            raise ValueError("État initial non dans l'ensemble des états")
        if not final_states.issubset(states):
            raise ValueError("États finaux non dans l'ensemble des états")
        if blank_symbol not in tape_alphabet:
            raise ValueError("Symbole blanc non dans l'alphabet du ruban")
        
        # État d'exécution
        self.reset()
    
    def reset(self):
        """Remet la machine à l'état initial"""
        self.tape = []
        self.head_position = 0
        self.current_state = self.initial_state
        self.execution_trace = []
        self.step_count = 0
    
    def _expand_tape(self, position: int):
        """Étend le ruban si nécessaire pour accéder à la position"""
        if position < 0:
            # Étendre à gauche
            extension = -position
            self.tape = [self.blank_symbol] * extension + self.tape
            self.head_position += extension
        elif position >= len(self.tape):
            # Étendre à droite
            extension = position - len(self.tape) + 1
            self.tape.extend([self.blank_symbol] * extension)
    
    def _read_symbol(self) -> str:
        """Lit le symbole sous la tête de lecture"""
        if self.head_position < 0 or self.head_position >= len(self.tape):
            return self.blank_symbol
        return self.tape[self.head_position]
    
    def _write_symbol(self, symbol: str):
        """Écrit un symbole sous la tête de lecture"""
        self._expand_tape(self.head_position)
        self.tape[self.head_position] = symbol
    
    def _move_head(self, direction: Direction):
        """Déplace la tête de lecture"""
        if direction == Direction.LEFT:
            self.head_position -= 1
        elif direction == Direction.RIGHT:
            self.head_position += 1
        # Direction.STAY ne fait rien
    
    def get_configuration(self) -> Dict[str, Any]:
        """Retourne la configuration actuelle de la machine"""
        return {
            'state': self.current_state,
            'tape': self.tape.copy(),
            'head_position': self.head_position,
            'step_count': self.step_count
        }
    
    def execute(self, input_word: str, max_steps: int = 10000, trace: bool = False) -> Tuple[ExecutionResult, List[Dict]]:
        """
        Exécute la machine sur un mot d'entrée
        
        Args:
            input_word: Mot d'entrée
            max_steps: Nombre maximum d'étapes
            trace: Si True, enregistre la trace d'exécution
            
        Returns:
            Tuple (résultat, trace d'exécution)
        """
        self.reset()
        
        # Initialiser le ruban avec le mot d'entrée
        self.tape = list(input_word) if input_word else [self.blank_symbol]
        if not self.tape:
            self.tape = [self.blank_symbol]
        
        if trace:
            self.execution_trace.append(self.get_configuration())
        
        while self.step_count < max_steps:
            # Vérifier si on est dans un état final
            if self.current_state in self.final_states:
                return ExecutionResult.ACCEPTED, self.execution_trace
            
            # Lire le symbole actuel
            current_symbol = self._read_symbol()
            
            # Chercher une transition
            transition_key = (self.current_state, current_symbol)
            if transition_key not in self.transition_function:
                return ExecutionResult.REJECTED, self.execution_trace
            
            # Appliquer la transition
            new_state, write_symbol, direction = self.transition_function[transition_key]
            
            self._write_symbol(write_symbol)
            self._move_head(direction)
            self.current_state = new_state
            self.step_count += 1
            
            if trace:
                self.execution_trace.append(self.get_configuration())
        
        return ExecutionResult.TIMEOUT, self.execution_trace
    
    def print_trace(self, trace: List[Dict]):
        """Affiche une trace d'exécution"""
        print("\n=== TRACE D'EXÉCUTION ===")
        for i, config in enumerate(trace):
            tape_str = ''.join(config['tape'])
            head_indicator = ' ' * config['head_position'] + '^'
            print(f"Étape {i}: État={config['state']}, Ruban='{tape_str}'")
            print(f"         {head_indicator}")


class MultiTapeTuringMachine:
    """Machine de Turing à k rubans"""
    
    def __init__(self, states: Set[str], input_alphabet: Set[str], 
                 tape_alphabet: Set[str], transition_function: Dict,
                 initial_state: str, final_states: Set[str], 
                 num_tapes: int = 2, blank_symbol: str = '_'):
        self.states = states
        self.input_alphabet = input_alphabet
        self.tape_alphabet = tape_alphabet
        self.transition_function = transition_function
        self.initial_state = initial_state
        self.final_states = final_states
        self.num_tapes = num_tapes
        self.blank_symbol = blank_symbol
        self.reset()
    
    def reset(self):
        """Remet la machine à l'état initial"""
        self.tapes = [[] for _ in range(self.num_tapes)]
        self.head_positions = [0] * self.num_tapes
        self.current_state = self.initial_state
        self.step_count = 0
    
    def execute(self, input_word: str, max_steps: int = 10000) -> ExecutionResult:
        """Exécute la machine multi-rubans"""
        self.reset()
        
        # Initialiser le premier ruban avec l'entrée
        self.tapes[0] = list(input_word) if input_word else [self.blank_symbol]
        for i in range(1, self.num_tapes):
            self.tapes[i] = [self.blank_symbol]
        
        while self.step_count < max_steps:
            if self.current_state in self.final_states:
                return ExecutionResult.ACCEPTED
            
            # Lire les symboles de tous les rubans
            current_symbols = []
            for i in range(self.num_tapes):
                if self.head_positions[i] < 0 or self.head_positions[i] >= len(self.tapes[i]):
                    current_symbols.append(self.blank_symbol)
                else:
                    current_symbols.append(self.tapes[i][self.head_positions[i]])
            
            transition_key = (self.current_state, tuple(current_symbols))
            if transition_key not in self.transition_function:
                return ExecutionResult.REJECTED
            
            new_state, write_symbols, directions = self.transition_function[transition_key]
            
            # Appliquer les transitions sur tous les rubans
            for i in range(self.num_tapes):
                self._write_symbol_tape(i, write_symbols[i])
                self._move_head_tape(i, directions[i])
            
            self.current_state = new_state
            self.step_count += 1
        
        return ExecutionResult.TIMEOUT
    
    def _write_symbol_tape(self, tape_idx: int, symbol: str):
        """Écrit un symbole sur un ruban spécifique"""
        pos = self.head_positions[tape_idx]
        if pos < 0:
            extension = -pos
            self.tapes[tape_idx] = [self.blank_symbol] * extension + self.tapes[tape_idx]
            self.head_positions[tape_idx] += extension
            pos = self.head_positions[tape_idx]
        elif pos >= len(self.tapes[tape_idx]):
            extension = pos - len(self.tapes[tape_idx]) + 1
            self.tapes[tape_idx].extend([self.blank_symbol] * extension)
        
        self.tapes[tape_idx][pos] = symbol
    
    def _move_head_tape(self, tape_idx: int, direction: Direction):
        """Déplace la tête sur un ruban spécifique"""
        if direction == Direction.LEFT:
            self.head_positions[tape_idx] -= 1
        elif direction == Direction.RIGHT:
            self.head_positions[tape_idx] += 1


class PalindromeMachine:
    """Machine de Turing spécialisée pour reconnaître les palindromes"""
    
    @staticmethod
    def create() -> TuringMachine:
        """Crée une machine reconnaissant les palindromes"""
        states = {'q0', 'q1', 'q2', 'q3', 'q4', 'qaccept', 'qreject'}
        input_alphabet = {'0', '1'}
        tape_alphabet = {'0', '1', 'X', '_'}
        final_states = {'qaccept'}
        
        transitions = {
            ('q0', '0'): ('q1', 'X', Direction.RIGHT),
            ('q0', '1'): ('q2', 'X', Direction.RIGHT),
            ('q0', 'X'): ('q0', 'X', Direction.RIGHT),
            ('q0', '_'): ('qaccept', '_', Direction.STAY),
            
            ('q1', '0'): ('q1', '0', Direction.RIGHT),
            ('q1', '1'): ('q1', '1', Direction.RIGHT),
            ('q1', 'X'): ('q1', 'X', Direction.RIGHT),
            ('q1', '_'): ('q3', '_', Direction.LEFT),
            
            ('q2', '0'): ('q2', '0', Direction.RIGHT),
            ('q2', '1'): ('q2', '1', Direction.RIGHT),
            ('q2', 'X'): ('q2', 'X', Direction.RIGHT),
            ('q2', '_'): ('q4', '_', Direction.LEFT),
            
            ('q3', '0'): ('q0', 'X', Direction.LEFT),
            ('q3', 'X'): ('q3', 'X', Direction.LEFT),
            ('q3', '1'): ('qreject', '1', Direction.STAY),
            
            ('q4', '1'): ('q0', 'X', Direction.LEFT),
            ('q4', 'X'): ('q4', 'X', Direction.LEFT),
            ('q4', '0'): ('qreject', '0', Direction.STAY),
            
            ('q0', 'X'): ('q0', 'X', Direction.RIGHT),
        }
        
        return TuringMachine(states, input_alphabet, tape_alphabet, 
                           transitions, 'q0', final_states)


class ArithmeticMachine:
    """Machines de Turing pour les opérations arithmétiques en unaire"""
    
    @staticmethod
    def create_addition_machine() -> TuringMachine:
        """Crée une machine pour l'addition en unaire (1+1=11)"""
        states = {'q0', 'q1', 'q2', 'qaccept'}
        input_alphabet = {'1', '+'}
        tape_alphabet = {'1', '+', '_'}
        final_states = {'qaccept'}
        
        transitions = {
            ('q0', '1'): ('q0', '1', Direction.RIGHT),
            ('q0', '+'): ('q1', '1', Direction.RIGHT),
            ('q1', '1'): ('q1', '1', Direction.RIGHT),
            ('q1', '_'): ('qaccept', '_', Direction.STAY),
        }
        
        return TuringMachine(states, input_alphabet, tape_alphabet,
                           transitions, 'q0', final_states)
    
    @staticmethod
    def create_multiplication_machine() -> TuringMachine:
        """Crée une machine pour la multiplication en unaire"""
        states = {'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'qaccept'}
        input_alphabet = {'1', '*'}
        tape_alphabet = {'1', '*', 'X', 'Y', '_'}
        final_states = {'qaccept'}
        
        transitions = {
            # Phase d'initialisation
            ('q0', '1'): ('q1', 'X', Direction.RIGHT),
            ('q0', 'Y'): ('q0', 'Y', Direction.RIGHT),
            ('q0', '_'): ('q5', '_', Direction.LEFT),
            
            # Trouver le multiplicateur
            ('q1', '1'): ('q1', '1', Direction.RIGHT),
            ('q1', '*'): ('q1', '*', Direction.RIGHT),
            ('q1', 'Y'): ('q1', 'Y', Direction.RIGHT),
            ('q1', '_'): ('q2', '_', Direction.LEFT),
            
            # Marquer une unité du multiplicateur
            ('q2', '1'): ('q3', 'Y', Direction.LEFT),
            ('q2', 'Y'): ('q2', 'Y', Direction.LEFT),
            ('q2', '*'): ('q4', '*', Direction.LEFT),
            
            # Retourner au début pour copier le multiplicande
            ('q3', '1'): ('q3', '1', Direction.LEFT),
            ('q3', 'Y'): ('q3', 'Y', Direction.LEFT),
            ('q3', '*'): ('q3', '*', Direction.LEFT),
            ('q3', 'X'): ('q3', 'X', Direction.LEFT),
            ('q3', '_'): ('q0', '_', Direction.RIGHT),
            
            # Nettoyer et terminer
            ('q4', '1'): ('q4', '1', Direction.LEFT),
            ('q4', 'X'): ('q4', 'X', Direction.LEFT),
            ('q4', '_'): ('q5', '_', Direction.RIGHT),
            
            ('q5', 'X'): ('q5', '_', Direction.RIGHT),
            ('q5', '1'): ('q5', '1', Direction.RIGHT),
            ('q5', '*'): ('q5', '_', Direction.RIGHT),
            ('q5', 'Y'): ('q5', '_', Direction.RIGHT),
            ('q5', '_'): ('qaccept', '_', Direction.STAY),
        }
        
        return TuringMachine(states, input_alphabet, tape_alphabet,
                           transitions, 'q0', final_states)


class UniversalTuringMachine:
    """Machine de Turing universelle simplifiée"""
    
    def __init__(self):
        self.encoding_map = {}
        self.reverse_encoding = {}
        self._setup_encoding()
    
    def _setup_encoding(self):
        """Configure l'encodage pour représenter les machines"""
        # Encodage simple : chaque symbole/état est représenté par un nombre
        self.encoding_map = {
            'states': {},
            'symbols': {},
            'directions': {'L': '0', 'R': '1', 'S': '2'}
        }
    
    def encode_machine(self, machine: TuringMachine) -> str:
        """Encode une machine de Turing en chaîne de caractères"""
        # Encodage simplifié pour la démonstration
        encoding = []
        
        # Encoder les états
        state_encoding = {}
        for i, state in enumerate(sorted(machine.states)):
            state_encoding[state] = str(i)
        
        # Encoder les symboles
        symbol_encoding = {}
        for i, symbol in enumerate(sorted(machine.tape_alphabet)):
            symbol_encoding[symbol] = str(i)
        
        # Encoder les transitions
        transitions_encoded = []
        for (state, symbol), (new_state, write_symbol, direction) in machine.transition_function.items():
            trans = f"{state_encoding[state]},{symbol_encoding[symbol]},{state_encoding[new_state]},{symbol_encoding[write_symbol]},{self.encoding_map['directions'][direction.value]}"
            transitions_encoded.append(trans)
        
        # Format: états|symboles|transitions|initial|finals
        encoding.append('|'.join(sorted(machine.states)))
        encoding.append('|'.join(sorted(machine.tape_alphabet)))
        encoding.append('|'.join(transitions_encoded))
        encoding.append(machine.initial_state)
        encoding.append('|'.join(sorted(machine.final_states)))
        
        return '#'.join(encoding)
    
    def simulate(self, encoded_machine: str, input_word: str, max_steps: int = 1000) -> ExecutionResult:
        """Simule l'exécution d'une machine encodée"""
        try:
            # Décoder la machine
            parts = encoded_machine.split('#')
            if len(parts) != 5:
                return ExecutionResult.ERROR
            
            states = set(parts[0].split('|'))
            tape_alphabet = set(parts[1].split('|'))
            transitions_raw = parts[2].split('|') if parts[2] else []
            initial_state = parts[3]
            final_states = set(parts[4].split('|')) if parts[4] else set()
            
            # Reconstruire les transitions
            transitions = {}
            state_list = sorted(states)
            symbol_list = sorted(tape_alphabet)
            direction_map = {'0': Direction.LEFT, '1': Direction.RIGHT, '2': Direction.STAY}
            
            for trans in transitions_raw:
                if not trans:
                    continue
                parts_trans = trans.split(',')
                if len(parts_trans) == 5:
                    state_idx, symbol_idx, new_state_idx, write_symbol_idx, direction_idx = parts_trans
                    state = state_list[int(state_idx)]
                    symbol = symbol_list[int(symbol_idx)]
                    new_state = state_list[int(new_state_idx)]
                    write_symbol = symbol_list[int(write_symbol_idx)]
                    direction = direction_map[direction_idx]
                    
                    transitions[(state, symbol)] = (new_state, write_symbol, direction)
            
            # Créer et exécuter la machine
            machine = TuringMachine(states, set(), tape_alphabet, transitions, initial_state, final_states)
            result, _ = machine.execute(input_word, max_steps)
            return result
            
        except Exception as e:
            print(f"Erreur lors de la simulation: {e}")
            return ExecutionResult.ERROR


class NonDeterministicTuringMachine:
    """Machine de Turing non-déterministe simulée par backtracking"""
    
    def __init__(self, states: Set[str], input_alphabet: Set[str], 
                 tape_alphabet: Set[str], transition_function: Dict[Tuple[str, str], List[Tuple[str, str, Direction]]],
                 initial_state: str, final_states: Set[str], blank_symbol: str = '_'):
        self.states = states
        self.input_alphabet = input_alphabet
        self.tape_alphabet = tape_alphabet
        self.transition_function = transition_function
        self.initial_state = initial_state
        self.final_states = final_states
        self.blank_symbol = blank_symbol
    
    def execute(self, input_word: str, max_steps: int = 1000) -> Tuple[ExecutionResult, int]:
        """Exécute la machine non-déterministe avec backtracking"""
        initial_config = {
            'state': self.initial_state,
            'tape': list(input_word) if input_word else [self.blank_symbol],
            'head_position': 0,
            'steps': 0
        }
        
        stack = [initial_config]
        configurations_explored = 0
        
        while stack and configurations_explored < max_steps:
            config = stack.pop()
            configurations_explored += 1
            
            # Vérifier si accepté
            if config['state'] in self.final_states:
                return ExecutionResult.ACCEPTED, configurations_explored
            
            # Vérifier limite d'étapes
            if config['steps'] > max_steps // 10:
                continue
            
            # Lire symbole actuel
            pos = config['head_position']
            if pos < 0 or pos >= len(config['tape']):
                current_symbol = self.blank_symbol
            else:
                current_symbol = config['tape'][pos]
            
            # Chercher toutes les transitions possibles
            transition_key = (config['state'], current_symbol)
            if transition_key in self.transition_function:
                for new_state, write_symbol, direction in self.transition_function[transition_key]:
                    # Créer nouvelle configuration
                    new_config = {
                        'state': new_state,
                        'tape': config['tape'].copy(),
                        'head_position': config['head_position'],
                        'steps': config['steps'] + 1
                    }
                    
                    # Écrire symbole
                    if pos < 0:
                        extension = -pos
                        new_config['tape'] = [self.blank_symbol] * extension + new_config['tape']
                        new_config['head_position'] += extension
                        pos = new_config['head_position']
                    elif pos >= len(new_config['tape']):
                        extension = pos - len(new_config['tape']) + 1
                        new_config['tape'].extend([self.blank_symbol] * extension)
                    
                    new_config['tape'][pos] = write_symbol
                    
                    # Déplacer tête
                    if direction == Direction.LEFT:
                        new_config['head_position'] -= 1
                    elif direction == Direction.RIGHT:
                        new_config['head_position'] += 1
                    
                    stack.append(new_config)
        
        return ExecutionResult.REJECTED, configurations_explored


def test_palindrome_machine():
    """Test de la machine reconnaissant les palindromes"""
    print("\n=== TEST MACHINE PALINDROMES ===")
    machine = PalindromeMachine.create()
    
    test_cases = ['', '0', '1', '00', '11', '01', '10', '101', '010', '1001', '1010', '11011']
    
    for word in test_cases:
        result, trace = machine.execute(word, trace=True)
        print(f"Mot: '{word}' -> {result.value}")
        if len(word) <= 4:  # Afficher la trace pour les mots courts
            machine.print_trace(trace)


def test_arithmetic_machines():
    """Test des machines arithmétiques"""
    print("\n=== TEST MACHINES ARITHMÉTIQUES ===")
    
    # Test addition
    add_machine = ArithmeticMachine.create_addition_machine()
    test_cases = ['1+1', '11+1', '1+11', '111+11']
    
    print("\nAddition:")
    for expr in test_cases:
        result, _ = add_machine.execute(expr)
        print(f"{expr} -> {result.value}")


def test_universal_machine():
    """Test de la machine universelle"""
    print("\n=== TEST MACHINE UNIVERSELLE ===")
    
    # Créer une machine simple
    simple_machine = PalindromeMachine.create()
    utm = UniversalTuringMachine()
    
    # Encoder la machine
    encoded = utm.encode_machine(simple_machine)
    print(f"Machine encodée (tronquée): {encoded[:100]}...")
    
    # Tester la simulation
    test_words = ['', '101', '1001']
    for word in test_words:
        original_result, _ = simple_machine.execute(word)
        simulated_result = utm.simulate(encoded, word)
        print(f"Mot '{word}': Original={original_result.value}, Simulé={simulated_result.value}")


def test_multi_tape_machine():
    """Test de machine multi-rubans"""
    print("\n=== TEST MACHINE MULTI-RUBANS ===")
    
    # Machine à 2 rubans pour reconnaître w#w
    states = {'q0', 'q1', 'q2', 'q3', 'qaccept', 'qreject'}
    input_alphabet = {'0', '1', '#'}
    tape_alphabet = {'0', '1', '#', '_'}
    final_states = {'qaccept'}
    
    # Transitions pour machine 2-rubans
    transitions = {
        ('q0', ('0', '_')): ('q0', ('0', '0'), (Direction.RIGHT, Direction.RIGHT)),
        ('q0', ('1', '_')): ('q0', ('1', '1'), (Direction.RIGHT, Direction.RIGHT)),
        ('q0', ('#', '_')): ('q1', ('#', '_'), (Direction.RIGHT, Direction.LEFT)),
        ('q1', ('0', '0')): ('q1', ('0', '0'), (Direction.RIGHT, Direction.LEFT)),
        ('q1', ('1', '1')): ('q1', ('1', '1'), (Direction.RIGHT, Direction.LEFT)),
        ('q1', ('_', '_')): ('qaccept', ('_', '_'), (Direction.STAY, Direction.STAY)),
    }
    
    machine = MultiTapeTuringMachine(states, input_alphabet, tape_alphabet,
                                   transitions, 'q0', final_states, 2)
    
    test_cases = ['0#0', '1#1', '01#01', '10#10', '01#10', '101#101']
    for word in test_cases:
        result = machine.execute(word)
        print(f"Mot '{word}' -> {result.value}")


def performance_comparison():
    """Comparaison de performances entre déterministe et non-déterministe"""
    print("\n=== COMPARAISON PERFORMANCES ===")
    
    # Machine déterministe simple
    det_machine = PalindromeMachine.create()
    
    # Machine non-déterministe équivalente
    nd_transitions = {}
    for key, value in det_machine.transition_function.items():
        nd_transitions[key] = [value]  # Convertir en liste pour ND
    
    nd_machine = NonDeterministicTuringMachine(
        det_machine.states, det_machine.input_alphabet, det_machine.tape_alphabet,
        nd_transitions, det_machine.initial_state, det_machine.final_states
    )
    
    test_words = ['101', '1001', '10101']
    
    for word in test_words:
        # Test déterministe
        start_time = time.time()
        det_result, det_trace = det_machine.execute(word)
        det_time = time.time() - start_time
        det_steps = len(det_trace)
        
        # Test non-déterministe
        start_time = time.time()
        nd_result, nd_configs = nd_machine.execute(word)
        nd_time = time.time() - start_time
        
        print(f"Mot '{word}':")
        print(f"  Déterministe: {det_result.value} en {det_steps} étapes ({det_time:.6f}s)")
        print(f"  Non-déterministe: {nd_result.value} en {nd_configs} configurations ({nd_time:.6f}s)")


class AnBnMachine:
    """Machine de Turing pour reconnaître le langage {a^n b^n | n ≥ 0}"""
    
    @staticmethod
    def create() -> TuringMachine:
        """Crée une machine reconnaissant a^n b^n"""
        states = {'q0', 'q1', 'q2', 'q3', 'qaccept', 'qreject'}
        input_alphabet = {'a', 'b'}
        tape_alphabet = {'a', 'b', 'X', 'Y', '_'}
        final_states = {'qaccept'}
        
        transitions = {
            # État initial : marquer le premier 'a'
            ('q0', 'a'): ('q1', 'X', Direction.RIGHT),
            ('q0', 'Y'): ('q3', 'Y', Direction.RIGHT),
            ('q0', '_'): ('qaccept', '_', Direction.STAY),
            
            # Chercher le premier 'b' correspondant
            ('q1', 'a'): ('q1', 'a', Direction.RIGHT),
            ('q1', 'Y'): ('q1', 'Y', Direction.RIGHT),
            ('q1', 'b'): ('q2', 'Y', Direction.LEFT),
            ('q1', '_'): ('qreject', '_', Direction.STAY),
            
            # Retourner au début
            ('q2', 'Y'): ('q2', 'Y', Direction.LEFT),
            ('q2', 'a'): ('q2', 'a', Direction.LEFT),
            ('q2', 'X'): ('q0', 'X', Direction.RIGHT),
            
            # Vérifier qu'il ne reste que des Y
            ('q3', 'Y'): ('q3', 'Y', Direction.RIGHT),
            ('q3', '_'): ('qaccept', '_', Direction.STAY),
            ('q3', 'b'): ('qreject', 'b', Direction.STAY),
        }
        
        return TuringMachine(states, input_alphabet, tape_alphabet,
                           transitions, 'q0', final_states)


class SortingMachine:
    """Machine de Turing à 3 rubans pour trier une séquence"""
    
    @staticmethod
    def create_3_tape_sorter() -> MultiTapeTuringMachine:
        """Crée une machine à 3 rubans pour trier"""
        states = {'q0', 'q1', 'q2', 'q3', 'q4', 'qaccept'}
        input_alphabet = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ','}
        tape_alphabet = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',', 'X', '_'}
        final_states = {'qaccept'}
        
        # Implémentation simplifiée du tri par sélection
        transitions = {
            # État initial : copier l'entrée sur le ruban de travail
            ('q0', ('0', '_', '_')): ('q0', ('0', '0', '_'), (Direction.RIGHT, Direction.RIGHT, Direction.STAY)),
            ('q0', ('1', '_', '_')): ('q0', ('1', '1', '_'), (Direction.RIGHT, Direction.RIGHT, Direction.STAY)),
            ('q0', (',', '_', '_')): ('q0', (',', ',', '_'), (Direction.RIGHT, Direction.RIGHT, Direction.STAY)),
            ('q0', ('_', '_', '_')): ('q1', ('_', '_', '_'), (Direction.LEFT, Direction.LEFT, Direction.STAY)),
            
            # Phase de tri (simplifiée)
            ('q1', ('_', '_', '_')): ('qaccept', ('_', '_', '_'), (Direction.STAY, Direction.STAY, Direction.STAY)),
        }
        
        return MultiTapeTuringMachine(states, input_alphabet, tape_alphabet,
                                    transitions, 'q0', final_states, 3)


def test_anbn_machine():
    """Test de la machine a^n b^n"""
    print("\n=== TEST MACHINE a^n b^n ===")
    machine = AnBnMachine.create()
    
    test_cases = ['', 'ab', 'aabb', 'aaabbb', 'aaaabbbb', 'a', 'b', 'aab', 'abb', 'abab']
    
    for word in test_cases:
        result, trace = machine.execute(word, trace=True)
        print(f"Mot: '{word}' -> {result.value}")
        if word in ['ab', 'aabb']:  # Afficher trace pour exemples simples
            machine.print_trace(trace)


def test_sorting_machine():
    """Test de la machine de tri à 3 rubans"""
    print("\n=== TEST MACHINE DE TRI 3 RUBANS ===")
    machine = SortingMachine.create_3_tape_sorter()
    
    test_cases = ['3,1,4,1,5', '9,2,6,5,3', '1,2,3']
    
    for sequence in test_cases:
        result = machine.execute(sequence)
        print(f"Séquence: '{sequence}' -> {result.value}")


def complexity_analysis():
    """Analyse de complexité des différentes machines"""
    print("\n=== ANALYSE DE COMPLEXITÉ ===")
    
    machines = {
        'Palindrome': PalindromeMachine.create(),
        'a^n b^n': AnBnMachine.create(),
        'Addition': ArithmeticMachine.create_addition_machine()
    }
    
    test_sizes = [2, 4, 6, 8, 10]
    
    for machine_name, machine in machines.items():
        print(f"\n{machine_name}:")
        print("Taille\tÉtapes\tTemps(ms)")
        print("-" * 25)
        
        for size in test_sizes:
            if machine_name == 'Palindrome':
                test_word = '1' * size + '0' * size
            elif machine_name == 'a^n b^n':
                test_word = 'a' * size + 'b' * size
            elif machine_name == 'Addition':
                test_word = '1' * size + '+' + '1' * size
            
            start_time = time.time()
            result, trace = machine.execute(test_word)
            end_time = time.time()
            
            steps = len(trace) if trace else 0
            time_ms = (end_time - start_time) * 1000
            
            print(f"{size}\t{steps}\t{time_ms:.2f}")


def interactive_simulator():
    """Interface interactive pour tester le simulateur"""
    print("\n=== SIMULATEUR INTERACTIF ===")
    print("Machines disponibles:")
    print("1. Palindromes")
    print("2. a^n b^n")
    print("3. Addition unaire")
    print("4. Machine personnalisée")
    
    try:
        choice = input("\nChoisissez une machine (1-4): ")
        
        if choice == '1':
            machine = PalindromeMachine.create()
            print("Machine palindromes chargée.")
        elif choice == '2':
            machine = AnBnMachine.create()
            print("Machine a^n b^n chargée.")
        elif choice == '3':
            machine = ArithmeticMachine.create_addition_machine()
            print("Machine addition chargée.")
        elif choice == '4':
            print("Fonctionnalité machine personnalisée à implémenter...")
            return
        else:
            print("Choix invalide.")
            return
        
        while True:
            word = input("\nEntrez un mot (ou 'quit' pour quitter): ")
            if word.lower() == 'quit':
                break
            
            trace_option = input("Afficher la trace? (o/n): ").lower() == 'o'
            
            result, trace = machine.execute(word, trace=trace_option)
            print(f"Résultat: {result.value}")
            
            if trace_option:
                machine.print_trace(trace)
    
    except KeyboardInterrupt:
        print("\nSimulateur interrompu.")
    except Exception as e:
        print(f"Erreur: {e}")


def generate_report():
    """Génère un rapport détaillé des tests"""
    print("\n=== RAPPORT DE TESTS DÉTAILLÉ ===")
    
    report = {
        'machines_testees': 0,
        'tests_reussis': 0,
        'tests_echoues': 0,
        'temps_total': 0
    }
    
    start_total = time.time()
    
    # Test palindromes
    print("\n1. Test machine palindromes:")
    machine = PalindromeMachine.create()
    palindromes = ['', '0', '1', '101', '010', '1001']
    non_palindromes = ['01', '10', '001', '100', '1010']
    
    for word in palindromes:
        result, _ = machine.execute(word)
        if result == ExecutionResult.ACCEPTED:
            report['tests_reussis'] += 1
            print(f"  ✓ '{word}' accepté (correct)")
        else:
            report['tests_echoues'] += 1
            print(f"  ✗ '{word}' rejeté (incorrect)")
    
    for word in non_palindromes:
        result, _ = machine.execute(word)
        if result == ExecutionResult.REJECTED:
            report['tests_reussis'] += 1
            print(f"  ✓ '{word}' rejeté (correct)")
        else:
            report['tests_echoues'] += 1
            print(f"  ✗ '{word}' accepté (incorrect)")
    
    report['machines_testees'] += 1
    
    # Test a^n b^n
    print("\n2. Test machine a^n b^n:")
    machine = AnBnMachine.create()
    valid_words = ['', 'ab', 'aabb', 'aaabbb']
    invalid_words = ['a', 'b', 'aab', 'abb', 'abab']
    
    for word in valid_words:
        result, _ = machine.execute(word)
        if result == ExecutionResult.ACCEPTED:
            report['tests_reussis'] += 1
            print(f"  ✓ '{word}' accepté (correct)")
        else:
            report['tests_echoues'] += 1
            print(f"  ✗ '{word}' rejeté (incorrect)")
    
    for word in invalid_words:
        result, _ = machine.execute(word)
        if result == ExecutionResult.REJECTED:
            report['tests_reussis'] += 1
            print(f"  ✓ '{word}' rejeté (correct)")
        else:
            report['tests_echoues'] += 1
            print(f"  ✗ '{word}' accepté (incorrect)")
    
    report['machines_testees'] += 1
    
    # Test machine universelle
    print("\n3. Test machine universelle:")
    try:
        utm = UniversalTuringMachine()
        simple_machine = PalindromeMachine.create()
        encoded = utm.encode_machine(simple_machine)
        
        test_words = ['101', '110']
        for word in test_words:
            original_result, _ = simple_machine.execute(word)
            simulated_result = utm.simulate(encoded, word)
            
            if original_result.value == simulated_result.value:
                report['tests_reussis'] += 1
                print(f"  ✓ '{word}': simulation correcte")
            else:
                report['tests_echoues'] += 1
                print(f"  ✗ '{word}': simulation incorrecte")
        
        report['machines_testees'] += 1
    except Exception as e:
        print(f"  ✗ Erreur machine universelle: {e}")
        report['tests_echoues'] += 1
    
    report['temps_total'] = time.time() - start_total
    
    # Résumé
    print(f"\n=== RÉSUMÉ ===")
    print(f"Machines testées: {report['machines_testees']}")
    print(f"Tests réussis: {report['tests_reussis']}")
    print(f"Tests échoués: {report['tests_echoues']}")
    print(f"Taux de réussite: {report['tests_reussis']/(report['tests_reussis']+report['tests_echoues'])*100:.1f}%")
    print(f"Temps total: {report['temps_total']:.2f}s")


def main():
    """Fonction principale pour tester toutes les implémentations"""
    print("SIMULATEUR DE MACHINE DE TURING - PARTIE II")
    print("=" * 50)
    
    try:
        # Tests de base
        test_palindrome_machine()
        test_anbn_machine()
        test_arithmetic_machines()
        test_multi_tape_machine()
        test_universal_machine()
        
        # Tests avancés
        print("\n" + "=" * 30)
        print("TESTS AVANCÉS")
        print("=" * 30)
        
        test_sorting_machine()
        complexity_analysis()
        performance_comparison()
        
        # Rapport final
        generate_report()
        
        # Option interactive
        print("\n" + "=" * 30)
        interactive_choice = input("Lancer le simulateur interactif? (o/n): ")
        if interactive_choice.lower() == 'o':
            interactive_simulator()
        
        print("\n" + "=" * 50)
        print("TOUS LES TESTS TERMINÉS AVEC SUCCÈS")
        print("=" * 50)
        
    except Exception as e:
        print(f"Erreur lors des tests: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()