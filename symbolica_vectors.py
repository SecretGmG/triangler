from symbolica import N, S, N

class SymbolicaLorenzVec:
    def __init__(self, symbols : list):
        self.symbols = symbols
    @staticmethod
    def from_name(name):
        symbols = [S(f'{name}_{i}')for i in range(4)]
        return SymbolicaLorenzVec(symbols)
    
    def t(self):
        return self.symbols[0]

    def spacial(self):
        return SymbolicaVec(self.symbols[1:])
    
    def norm(self):
        return self * (1/self.squared()**(N(1)/2))
    
    def squared(self):
        return self*self
    
    @staticmethod
    def zero():
        return SymbolicaLorenzVec([N(0),N(0),N(0),N(0)])
    
    def get_subs_dict(self, values):
        return {sym : N(num) for sym, num in zip(self.symbols, values)}
    
    def __add__(self, other):
        if isinstance(other, SymbolicaLorenzVec):
            return SymbolicaLorenzVec([a+b for a, b in zip(self.symbols, other.symbols)])
        return [a+other for a in self.symbols]
    def __sub__(self, other):
        if isinstance(other, SymbolicaLorenzVec):
            return SymbolicaLorenzVec([a-b for a, b in zip(self.symbols, other.symbols)])
        return SymbolicaLorenzVec([a+other for a in self.symbols])
    def __neg__(self):
        return SymbolicaLorenzVec.zero() - self

    def __mul__(self, other):
        if isinstance(other, SymbolicaLorenzVec):
            return self.symbols[0]*other.symbols[0]-self.spacial()*other.spacial()
        return SymbolicaLorenzVec([a*other for a in self.symbols])
    def __rmul__(self, other):
        return self * other
    def __str__(self):
        return str(self.symbols)
    def __repr__(self):
        return str(self)

class SymbolicaVec:
    def __init__(self, symbols : list):
        self.symbols = symbols
    @staticmethod
    def from_name(name):
        values = [S(f'{name}_{i}')for i in range(1,4)]
        return SymbolicaVec(values)
    
    @staticmethod
    def zero():
        return SymbolicaVec([N(0),N(0),N(0)])
    
        
    def norm(self):
        return self * (1/self.squared()**(N(1)/2))
    
    def squared(self):
        return self*self

    def __add__(self, other):
        if isinstance(other, SymbolicaVec):
            return SymbolicaVec([a+b for a, b in zip(self.symbols, other.symbols)])
        return SymbolicaVec([a+other for a in self.symbols])
    def __sub__(self, other):
        if isinstance(other, SymbolicaVec):
            return SymbolicaVec([a-b for a, b in zip(self.symbols, other.symbols)])
        return SymbolicaVec([a-other for a in self.symbols])
    def __mul__(self, other):
        if isinstance(other, SymbolicaVec):
            return sum(a*b for a, b in zip(self.symbols, other.symbols))
        return SymbolicaVec([a*other for a in self.symbols])
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return SymbolicaVec.zero() - self
    def __str__(self):
        return str(self.symbols)
    def __repr__(self):
        return str(self)