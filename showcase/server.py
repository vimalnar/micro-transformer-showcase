from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from .core import ROOT, Session
from .devices import Simulation


def register_model(app, session, prefix, simulation=None):
    router = APIRouter(prefix=prefix)

    @router.get('/status')
    def status():
        return session.status()

    @router.post('/predict')
    def predict(body: dict):
        try:
            return session.predict(body['prompt'])
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(400, str(exc))

    @router.post('/train')
    def train(body: dict):
        try:
            session.start({key: body[key] for key in ('text','rounds','rate')})
            return {'started': True}
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(400, str(exc))

    @router.post('/pause')
    def pause():
        session.pause.set()
        return {'pause_requested':True}

    @router.put('/draft')
    def draft(body: dict):
        try:
            if not isinstance(body['text'],str) or len(body['text']) > 100000:
                raise ValueError('Draft too large or invalid')
            if type(body['rounds']) is not int or not 1 <= body['rounds'] <= 20 or body['rate'] not in (.00003,.0001,.0003):
                raise ValueError('Draft needs 1–20 rounds and a supported learning rate')
            with session.lock:
                session.draft = {key: body[key] for key in ('text','rounds','rate')}
                session.save()
            return {'saved':True}
        except (ValueError,KeyError) as exc:
            raise HTTPException(400,str(exc))

    @router.get('/practice')
    def practice():
        return FileResponse(ROOT / 'data' / session.scope / 'practice.json')

    @router.get('/checkpoint')
    def checkpoint():
        with session.lock:
            session.save()
            return Response(session.path.read_bytes(), media_type='application/octet-stream',
                            headers={'Content-Disposition':f'attachment; filename="{session.scope}.pt"'})

    if simulation is not None:
        @router.get('/simulation')
        def simulation_state():
            return simulation.snapshot()

        @router.post('/events')
        def send_event(body: dict):
            try:
                return simulation.send(body['place'],body['delivered'])
            except (ValueError,KeyError,TypeError) as exc:
                raise HTTPException(400,str(exc))
            except Exception as exc:
                raise HTTPException(500,f'Device prediction failed: {exc}')

        @router.post('/interpret')
        def interpret():
            try:
                return simulation.interpret()
            except ValueError as exc:
                raise HTTPException(400,str(exc))
            except Exception as exc:
                raise HTTPException(500,f'Device prediction failed: {exc}')

        @router.post('/clear')
        def clear():
            return simulation.clear()

    app.include_router(router)


def create_app(directory=None, device_directory=None, ava_directory=None):
    chatbot = Session(directory)
    devices = Session(device_directory or (Path(directory)/'devices' if directory else None),scope='devices')
    simulation = Simulation(devices)
    from .ava.session import AvaSession
    from .ava.api import register_ava
    ava = AvaSession(ava_directory or (Path(directory)/"ava" if directory else None))

    @asynccontextmanager
    async def lifespan(app):
        yield
        import asyncio
        for session in (chatbot,devices,ava):
            session.pause.set()
        while chatbot.busy or devices.busy or ava.busy:
            await asyncio.sleep(.05)

    app = FastAPI(lifespan=lifespan)
    app.state.session = chatbot  # Preserve existing Chatbot API consumers.
    app.state.sessions = {'chatbot':chatbot,'devices':devices,'ava':ava}
    app.state.simulation = simulation

    @app.middleware('http')
    async def local_only(request: Request, call_next):
        from starlette.responses import JSONResponse
        host = request.headers.get('host','').split(':')[0]
        origin = request.headers.get('origin')
        if host not in ('127.0.0.1','localhost','testserver') or (origin and origin != str(request.base_url).rstrip('/')):
            return JSONResponse({'detail':'Local same-origin requests only'}, status_code=403)
        return await call_next(request)

    register_ava(app,ava)
    register_model(app,chatbot,'/api')
    register_model(app,devices,'/api/devices',simulation)
    app.mount('/', StaticFiles(directory=ROOT/'web',html=True), name='web')
    return app
