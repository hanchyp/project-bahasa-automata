import re
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from base64 import b64encode
import matplotlib.pyplot as plt
import networkx as nx
from dfa import DFA
from io import BytesIO


app = Flask(__name__)
CORS(app)

# ======== REGEX to NFA =========

class State:
    _id = 0
    def __init__(self):
        self.id = State._id
        State._id += 1
        self.transitions = {}

    def add_transition(self, symbol, state):
        if symbol in self.transitions:
            self.transitions[symbol].add(state)
        else:
            self.transitions[symbol] = {state}

class NFA:
    def __init__(self, start, accept):
        self.start = start
        self.accept = accept

    def epsilon_closure(self, states):
        stack = list(states)
        closure = set(states)
        while stack:
            state = stack.pop()
            for next_state in state.transitions.get('ε', []):
                if next_state not in closure:
                    closure.add(next_state)
                    stack.append(next_state)
        return closure

    def simulate(self, string):
        current_states = self.epsilon_closure({self.start})
        for symbol in string:
            next_states = set()
            for state in current_states:
                for dest in state.transitions.get(symbol, []):
                    next_states |= self.epsilon_closure({dest})
            current_states = next_states
        return self.accept in current_states

def expand_ranges(regex):
    def expand(match):
        start, end = match.group(1), match.group(2)
        return '(' + '|'.join(chr(c) for c in range(ord(start), ord(end) + 1)) + ')'
    return re.sub(r'\[([a-zA-Z0-9])-([a-zA-Z0-9])\]', expand, regex)

def regex_to_nfa(regex):
    regex = expand_ranges(regex)

    def precedence(op):
        return {'*': 3, '.': 2, '|': 1}.get(op, 0)

    def to_postfix(regex):
        output = ''
        stack = []
        prev = None
        for char in regex:
            if char.isalnum():
                if prev and (prev.isalnum() or prev in {')', '*'}):
                    while stack and precedence('.') <= precedence(stack[-1]):
                        output += stack.pop()
                    stack.append('.')
                output += char
            elif char == '(':
                if prev and (prev.isalnum() or prev in {')', '*'}):
                    while stack and precedence('.') <= precedence(stack[-1]):
                        output += stack.pop()
                    stack.append('.')
                stack.append(char)
            elif char == ')':
                while stack and stack[-1] != '(':
                    output += stack.pop()
                stack.pop()
            elif char in {'*', '|'}:
                while stack and precedence(char) <= precedence(stack[-1]):
                    output += stack.pop()
                stack.append(char)
            prev = char
        while stack:
            output += stack.pop()
        return output

    def build(postfix):
        stack = []
        for char in postfix:
            if char.isalnum():
                s0, s1 = State(), State()
                s0.add_transition(char, s1)
                stack.append(NFA(s0, s1))
            elif char == '.':
                n2 = stack.pop()
                n1 = stack.pop()
                n1.accept.add_transition('ε', n2.start)
                stack.append(NFA(n1.start, n2.accept))
            elif char == '|':
                n2 = stack.pop()
                n1 = stack.pop()
                s0, s1 = State(), State()
                s0.add_transition('ε', n1.start)
                s0.add_transition('ε', n2.start)
                n1.accept.add_transition('ε', s1)
                n2.accept.add_transition('ε', s1)
                stack.append(NFA(s0, s1))
            elif char == '*':
                n = stack.pop()
                s0, s1 = State(), State()
                s0.add_transition('ε', n.start)
                s0.add_transition('ε', s1)
                n.accept.add_transition('ε', s1)
                n.accept.add_transition('ε', n.start)
                stack.append(NFA(s0, s1))
        return stack.pop()

    postfix = to_postfix(regex)
    return build(postfix)

@app.route('/test_regex', methods=['POST'])
def test_regex():
    data = request.json
    regex = data.get('regex')
    test_string = data.get('string')

    if not regex or test_string is None:
        return jsonify({'error': 'Mohon berikan regex dan string'}), 400

    try:
        nfa = regex_to_nfa(regex)
        accepted = nfa.simulate(test_string)

        return jsonify({
            'accepted': accepted
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ======== DFA EQUIVALENCE =========

def are_dfa_equivalent(dfa1, dfa2):
    """
    Fungsi untuk mengecek apakah dua DFA ekuivalen
    """
    if set(dfa1['alphabet']) != set(dfa2['alphabet']):
        return False, "Alfabet tidak sama, maka DFA tidak ekuivalen."
    
    states1 = dfa1['states']
    states2 = dfa2['states']
    pairs = [(s1, s2) for s1 in states1 for s2 in states2]
    
    marked = set()
    unmarked = set(pairs)
    
    # Mark pairs where one is final and other is not
    for s1, s2 in pairs:
        if (s1 in dfa1['final_state']) != (s2 in dfa2['final_state']):
            marked.add((s1, s2))
            unmarked.discard((s1, s2))
    
    # Iteratively mark pairs
    while True:
        new_marked = set()
        for s1, s2 in unmarked:
            for symbol in dfa1['alphabet']:
                next_s1 = dfa1['transitions'][s1][symbol]
                next_s2 = dfa2['transitions'][s2][symbol]
                if (next_s1, next_s2) in marked:
                    new_marked.add((s1, s2))
                    break
        
        if not new_marked:
            break
        
        marked.update(new_marked)
        unmarked.difference_update(new_marked)
    
    # Check if start states are equivalent
    if (dfa1['start_state'], dfa2['start_state']) in marked:
        return False, "State awal tidak ekuivalen, maka DFA tidak ekuivalen."
    
    return True, "DFA ekuivalen."

@app.route('/check_equivalence', methods=['POST'])
def check_equivalence():
    try:
        data = request.json
        dfa1 = data['dfa1']
        dfa2 = data['dfa2']
        
        # Validate DFA input
        for dfa in [dfa1, dfa2]:
            if not all(key in dfa for key in ['states', 'alphabet', 'transitions', 'start_state', 'final_state']):
                return jsonify({'error': 'DFA tidak lengkap'}), 400
        
        equivalent, message = are_dfa_equivalent(dfa1, dfa2)
        
        return jsonify({
            'equivalent': equivalent,
            'message': message
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ======== DFA STRING TEST =========
@app.route('/check_string', methods=['POST'])
def check_string():
    try:
        data = request.get_json()
        dfa = data.get('dfa')
        input_string = data.get('input_string')

        if not dfa or input_string is None:
            return jsonify({'error': 'Data tidak lengkap'}), 400

        # Simulasi DFA dengan langkah-langkah detail
        simulation_result = simulate_dfa_with_steps(dfa, input_string)
        
        return jsonify({
            'accepted': simulation_result['accepted'],
            'message': f'String "{input_string}" {"diterima" if simulation_result["accepted"] else "ditolak"} oleh DFA',
            'steps': simulation_result['steps'],
            'error': simulation_result.get('error')
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def simulate_dfa_with_steps(dfa, input_string):
    """
    Simulasi DFA dengan tracking setiap langkah
    """
    current_state = dfa['start_state']
    transitions = dfa['transitions']
    alphabet = dfa['alphabet']
    steps = []
    
    # Step awal
    steps.append({
        'step': 0,
        'symbol': '',
        'current_state': current_state,
        'next_state': current_state,
        'action': 'Start'
    })
    
    # Proses setiap karakter
    for i, symbol in enumerate(input_string):
        # Cek apakah symbol ada dalam alphabet
        if symbol not in alphabet:
            return {
                'accepted': False,
                'steps': steps,
                'error': f"Symbol '{symbol}' tidak ada dalam alphabet"
            }
        
        # Cek apakah ada transisi untuk state dan symbol saat ini
        if current_state not in transitions or symbol not in transitions[current_state]:
            return {
                'accepted': False,
                'steps': steps,
                'error': f"Tidak ada transisi dari state '{current_state}' dengan symbol '{symbol}'"
            }
        
        # Lakukan transisi
        next_state = transitions[current_state][symbol]
        steps.append({
            'step': i + 1,
            'symbol': symbol,
            'current_state': current_state,
            'next_state': next_state,
            'action': f"Read '{symbol}'"
        })
        
        current_state = next_state
    
    # Cek apakah state akhir adalah final state
    accepted = current_state in dfa['final_state']
    
    return {
        'accepted': accepted,
        'steps': steps,
        'final_state': current_state
    }

def simulate_dfa(dfa, input_string):
    """
    Simulasi DFA sederhana (untuk kompatibilitas mundur)
    """
    current = dfa['start_state']
    transitions = dfa['transitions']
    alphabet = dfa['alphabet']

    for symbol in input_string:
        if symbol not in alphabet:
            return False
        if symbol not in transitions.get(current, {}):
            return False
        current = transitions[current][symbol]

    return current in dfa['final_state']

def draw_dfa(dfa):
    """
    Menggambar visualisasi DFA menggunakan NetworkX dan Matplotlib
    """
    G = nx.MultiDiGraph()

    states = dfa['states']
    alphabet = dfa['alphabet']
    start_state = dfa['start_state']
    final_states = dfa['final_state']
    transitions = dfa['transitions']

    # Tambahkan nodes
    for state in states:
        G.add_node(state)

    # Tambahkan edges dengan label
    for state in states:
        for symbol in alphabet:
            next_state = transitions.get(state, {}).get(symbol)
            if next_state:
                G.add_edge(state, next_state, label=symbol)

    # Layout untuk positioning nodes
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Buat figure
    plt.figure(figsize=(10, 8))
    plt.clf()
    
    # Warna nodes: hijau untuk final states, biru untuk yang lain
    node_colors = ['lightgreen' if s in final_states else 'lightblue' for s in G.nodes]
    
    # Gambar nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.9)
    
    # Gambar labels untuk nodes
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    # Tandai start state dengan panah dari luar
    start_pos = pos[start_state]
    plt.annotate("START", xy=start_pos, xytext=(start_pos[0]-0.3, start_pos[1]+0.2),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=2),
                 fontsize=10, fontweight='bold', color='red')

    # Gabungkan edge labels jika ada multiple symbols pada edge yang sama
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        edge_key = (u, v)
        if edge_key in edge_labels:
            edge_labels[edge_key] += ',' + data['label']
        else:
            edge_labels[edge_key] = data['label']

    # Gambar edges
    nx.draw_networkx_edges(G, pos, connectionstyle='arc3, rad=0.1', 
                          arrowsize=20, edge_color='gray', width=1.5)
    
    # Gambar edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    # Tambahkan double circle untuk final states
    for state in final_states:
        if state in pos:
            circle = plt.Circle(pos[state], 0.15, fill=False, color='green', linewidth=3)
            plt.gca().add_patch(circle)

    plt.title("DFA Visualization", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    # Konversi ke base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

@app.route('/visualize_dfa', methods=['POST'])
def visualize_dfa():
    try:
        data = request.get_json()
        dfa = data.get('dfa')
        if not dfa:
            return jsonify({'error': 'DFA data tidak ditemukan'}), 400

        img_base64 = draw_dfa(dfa)
        return jsonify({'image': img_base64})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

# ========= DFA MINIMIZATION =========
class DFAMinimizer:
    def __init__(self, states, alphabet, transitions, start_state, accept_states):
        self.states = set(states)
        self.alphabet = set(alphabet)
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = set(accept_states)
        
    def minimize(self):
        """Minimasi DFA menggunakan algoritma pengelompokan state ekuivalen"""
        # Langkah 1: Hapus state yang tidak dapat dijangkau
        reachable_states = self._get_reachable_states()
        
        # Langkah 2: Partisi awal berdasarkan accepting dan non-accepting states
        accepting = set(self.accept_states) & reachable_states
        non_accepting = reachable_states - accepting
        
        partitions = []
        if non_accepting:
            partitions.append(non_accepting)
        if accepting:
            partitions.append(accepting)
            
        # Langkah 3: Refine partitions sampai tidak ada perubahan
        changed = True
        while changed:
            changed = False
            new_partitions = []
            
            for partition in partitions:
                sub_partitions = self._refine_partition(partition, partitions)
                if len(sub_partitions) > 1:
                    changed = True
                new_partitions.extend(sub_partitions)
                
            partitions = new_partitions
            
        # Langkah 4: Buat DFA minimal
        return self._build_minimal_dfa(partitions, reachable_states)
    
    def _get_reachable_states(self):
        """Dapatkan semua state yang dapat dijangkau dari start state"""
        reachable = set()
        stack = [self.start_state]
        
        while stack:
            state = stack.pop()
            if state in reachable:
                continue
                
            reachable.add(state)
            
            for symbol in self.alphabet:
                if (state, symbol) in self.transitions:
                    next_state = self.transitions[(state, symbol)]
                    if next_state not in reachable:
                        stack.append(next_state)
                        
        return reachable
    
    def _refine_partition(self, partition, all_partitions):
        """Refine sebuah partisi berdasarkan transisi"""
        if len(partition) <= 1:
            return [partition]
            
        # Buat dictionary untuk mengelompokkan state berdasarkan signature transisi
        signature_groups = {}
        
        for state in partition:
            signature = []
            for symbol in sorted(self.alphabet):
                if (state, symbol) in self.transitions:
                    next_state = self.transitions[(state, symbol)]
                    # Cari partisi mana yang mengandung next_state
                    for i, part in enumerate(all_partitions):
                        if next_state in part:
                            signature.append(i)
                            break
                else:
                    signature.append(-1)  # Tidak ada transisi
                    
            signature_tuple = tuple(signature)
            if signature_tuple not in signature_groups:
                signature_groups[signature_tuple] = set()
            signature_groups[signature_tuple].add(state)
            
        return list(signature_groups.values())
    
    def _build_minimal_dfa(self, partitions, reachable_states):
        """Bangun DFA minimal dari partisi"""
        # Buat mapping dari state asli ke representatif partisi
        state_to_partition = {}
        partition_representatives = {}
        
        for i, partition in enumerate(partitions):
            rep = f"q{i}"
            partition_representatives[rep] = partition
            for state in partition:
                state_to_partition[state] = rep
                
        # State minimal
        minimal_states = list(partition_representatives.keys())
        
        # Start state minimal
        minimal_start = state_to_partition[self.start_state]
        
        # Accept states minimal
        minimal_accept = []
        for rep, partition in partition_representatives.items():
            if any(state in self.accept_states for state in partition):
                minimal_accept.append(rep)
                
        # Transisi minimal
        minimal_transitions = {}
        for rep, partition in partition_representatives.items():
            # Ambil state pertama dari partisi sebagai representatif
            representative_state = next(iter(partition))
            
            for symbol in self.alphabet:
                if (representative_state, symbol) in self.transitions:
                    next_state = self.transitions[(representative_state, symbol)]
                    if next_state in state_to_partition:
                        minimal_transitions[(rep, symbol)] = state_to_partition[next_state]
                        
        return {
            'states': minimal_states,
            'alphabet': list(self.alphabet),
            'transitions': {f"{k[0]},{k[1]}": v for k, v in minimal_transitions.items()},
            'start_state': minimal_start,
            'accept_states': minimal_accept,
            'partition_mapping': {rep: list(partition) for rep, partition in partition_representatives.items()}
        }

@app.route('/minimize', methods=['POST'])
def minimize_dfa():
    try:
        data = request.json
        
        states = data.get('states', [])
        alphabet = data.get('alphabet', [])
        transitions = data.get('transitions', {})
        start_state = data.get('start_state', '')
        accept_states = data.get('accept_states', [])
        
        # Validasi input
        if not states or not alphabet or not start_state:
            return jsonify({'error': 'Input tidak lengkap'}), 400
            
        # Konversi format transisi
        trans_dict = {}
        for key, value in transitions.items():
            if ',' in key:
                state, symbol = key.split(',', 1)
                trans_dict[(state.strip(), symbol.strip())] = value.strip()
                
        # Buat dan minimasi DFA
        dfa = DFAMinimizer(states, alphabet, trans_dict, start_state, accept_states)
        minimal_dfa = dfa.minimize()
        
        return jsonify({
            'success': True,
            'original': {
                'states': states,
                'alphabet': alphabet,
                'transitions': transitions,
                'start_state': start_state,
                'accept_states': accept_states
            },
            'minimal': minimal_dfa
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ROUTING HALAMANNYA (PAGES)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dfa-equivalence')
def dfa_equivalence():
    return render_template('dfa-equivalence.html')

@app.route('/regex-to-nfa')
def regex_to_nfa_page():
    return render_template('regex-to-nfa.html')

@app.route('/dfa-string-test')
def dfa_string_test():
    return render_template('dfa-string-test.html')

@app.route('/dfa-minimization')
def dfa_minimization():
    return render_template('dfa-minimization.html')

if __name__ == '__main__':
    app.run(debug=True, port=8080)