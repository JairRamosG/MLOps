from fastapi import FastAPI

MLOpsDataPath = FastAPI()

@MLOpsDataPath.get('/')
def serverGet():
    #return 'Hola DataPath!'
    return {'mensaje': 'Hola DataPath!'}

@MLOpsDataPath.get('/modelo/{id}') #El id puede ser cualquier cosa, nosotros usaremos nombres de Modelos
def serverGet(id):
    #return 'Hola DataPath!'
    return {'mensaje': f'Modelo con id {id}'}

@MLOpsDataPath.get('/modelo/entero/{id}') #El id solo puede ser un numero enterosa, nosotros usaremos nombres de Modelos
def serverGet(id: int):
    #return 'Hola DataPath!'
    return {'mensaje': f'Modelo con id {id}'}

@MLOpsDataPath.post('/metodoPost')
def serverPost():
    return 'Hola'

#uvicorn programa-api:MLOpsDataPath --reload