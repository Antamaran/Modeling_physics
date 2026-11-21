class World:
    def __init__(self):
        self.bodies=[]

    def step(self):
        for body in self.bodies:
            body.process()
        
        