class TriangularFuzzyNumber:

    # Initialise with lower, most likely, and upper bound (l,m,u)
    def __init__(self, l:float, m:float, u:float):
        self.l = l
        self.m = m
        self.u = u

    # Fuzzy addition
    def __add__(self, other):

    
    # Fuzzy multiplication
    def __mul__(self, other):

    # Fuzzy division
    def __truediv__(self, other):

    # Reciprocals
    def reciprocal(self):

    # Defuzzification with the centroid method
    def defuzzify(self, method='centroid'):


    # Fuzzy distance
    def distance_to(self, other, method='vertex'):


    @staticmethod
    # Linguistics
    def from_linguistic(term: str):
    

    def __repr__(self):
        return f"TFN({self.l:.3f}, {self.m:.3f}, {self.u:.3f})"