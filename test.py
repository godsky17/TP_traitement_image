import time
from app import (
    PalindromeMachine,
    AnBnMachine,
    ArithmeticMachine,
    MultiTapeTuringMachine,
    ExecutionResult,
    Direction
)

def test_machine(name, machine, test_cases, is_multi=False, is_nondeterministic=False):
    print(f"\n=== TEST: {name} ===")
    print(f"Type: {'Non-déterministe' if is_nondeterministic else 'Déterministe'}")

    total_steps = 0
    total_time = 0
    successes = 0
    total = len(test_cases)

    for word, expected in test_cases:
        start = time.time()

        if is_multi:
            result = machine.execute(word)
            steps = machine.step_count  # MultiTapeTuringMachine has step_count
        else:
            result, trace = machine.execute(word, trace=True)
            steps = len(trace)

        end = time.time()
        duration = (end - start) * 1000  # ms

        ok = result == expected
        total_steps += steps
        total_time += duration
        status = "✓" if ok else "✗"
        print(f"{status} '{word}' => {result.value} | Étapes: {steps}, Temps: {duration:.2f} ms (attendu: {expected.value})")

        if ok:
            successes += 1

    avg_steps = total_steps / total if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0
    print(f"Résumé : {successes}/{total} réussis | Étapes moyennes: {avg_steps:.1f} | Temps moyen: {avg_time:.2f} ms")


def test_palindrome():
    machine = PalindromeMachine.create()
    tests = [
        ("", ExecutionResult.ACCEPTED),
        ("0", ExecutionResult.ACCEPTED),
        ("01", ExecutionResult.REJECTED),
        ("101", ExecutionResult.ACCEPTED),
        ("100", ExecutionResult.REJECTED)
    ]
    test_machine("Palindromes", machine, tests)


def test_anbn():
    machine = AnBnMachine.create()
    tests = [
        ("", ExecutionResult.ACCEPTED),
        ("ab", ExecutionResult.ACCEPTED),
        ("aabb", ExecutionResult.ACCEPTED),
        ("aaabbb", ExecutionResult.ACCEPTED),
        ("aab", ExecutionResult.REJECTED),
        ("abab", ExecutionResult.REJECTED),
    ]
    test_machine("a^n b^n", machine, tests)


def test_addition():
    machine = ArithmeticMachine.create_addition_machine()
    tests = [
        ("1+1", ExecutionResult.ACCEPTED),
        ("11+1", ExecutionResult.ACCEPTED),
        ("1+11", ExecutionResult.ACCEPTED),
        ("111+111", ExecutionResult.ACCEPTED),
        ("1111", ExecutionResult.REJECTED),
    ]
    test_machine("Addition unaire", machine, tests)


def test_multitape():
    # Machine 2 rubans pour w#w
    states = {'q0', 'q1', 'q2', 'q3', 'qaccept', 'qreject'}
    input_alphabet = {'0', '1', '#'}
    tape_alphabet = {'0', '1', '#', '_'}
    final_states = {'qaccept'}

    transitions = {
        ('q0', ('0', '_')): ('q0', ('0', '0'), (Direction.RIGHT, Direction.RIGHT)),
        ('q0', ('1', '_')): ('q0', ('1', '1'), (Direction.RIGHT, Direction.RIGHT)),
        ('q0', ('#', '_')): ('q1', ('#', '_'), (Direction.RIGHT, Direction.LEFT)),
        ('q1', ('0', '0')): ('q1', ('0', '0'), (Direction.RIGHT, Direction.LEFT)),
        ('q1', ('1', '1')): ('q1', ('1', '1'), (Direction.RIGHT, Direction.LEFT)),
        ('q1', ('_', '_')): ('qaccept', ('_', '_'), (Direction.STAY, Direction.STAY)),
    }

    machine = MultiTapeTuringMachine(states, input_alphabet, tape_alphabet,
                                     transitions, 'q0', final_states, num_tapes=2)

    tests = [
        ("01#01", ExecutionResult.ACCEPTED),
        ("10#10", ExecutionResult.ACCEPTED),
        ("10#01", ExecutionResult.REJECTED),
        ("111#111", ExecutionResult.ACCEPTED),
    ]
    test_machine("w#w (2 rubans)", machine, tests, is_multi=True)


if __name__ == "__main__":
    test_palindrome()
    test_anbn()
    test_addition()
    test_multitape()
