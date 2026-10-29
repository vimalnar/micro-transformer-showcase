from fastapi import APIRouter,HTTPException
from fastapi.responses import Response
from . import world as W


def register_ava(app,session):
    router=APIRouter(prefix='/api/ava')
    def perform(fn,*args,**kwargs):
        try:return fn(*args,**kwargs)
        except (ValueError,KeyError,TypeError) as exc:raise HTTPException(400,str(exc))

    @router.get('/status')
    def status():return session.snapshot()

    @router.post('/step')
    def step():return perform(session.action)

    @router.post('/retry')
    def retry():return perform(session.retry)

    @router.post('/another')
    def another():return perform(session.retry,True)

    @router.post('/world')
    def world(body:dict):
        with session.lock:
            perform(session.idle)
            try:
                if 'preset' in body:
                    if body['preset'] not in ('empty','detour','zigzag','locks') or type(body.get('large',False)) is not bool:raise ValueError('Unsupported preset')
                    if body['preset']=='locks' and not body.get('large'):raise ValueError('Keys and doors require expanded encoding')
                    w=W.preset(body['preset'],body.get('large',False))
                elif body.get('expand') is True:w=W.expand(session.layout)
                elif 'direction' in body:
                    w=W.fresh(session.layout);w['agent']['direction']=body['direction']
                else:w=W.edit(session.layout,body['x'],body['y'],body['tool'])
                return session.apply_layout(w)
            except (ValueError,KeyError,TypeError) as exc:raise HTTPException(400,str(exc))

    @router.post('/train')
    def train(body:dict):
        perform(session.start,body.get('settings',{}),body.get('selection'))
        return {'started':True}

    @router.post('/settings')
    def settings(body:dict):
        return perform(session.configure_practice,body.get('settings',{}),body.get('selection'))

    @router.post('/pause')
    def pause():session.pause.set();return {'pause_requested':True}

    @router.post('/check')
    def check():perform(session.check);return {'started':True}

    @router.get('/examples')
    def examples():
        with session.lock:
            data,worlds=perform(W.dataset,session.layout,session.selection,session.model.has_memory)
            return dict(total=len(data),worlds=len(worlds),examples=[dict(id=e['id'],tokens=W.tokens(e['features']),label=W.ACTIONS[e['label']],memory=e['memory']) for e in data[:20]],scope='Generated supervised practice; labels never used for Playground actions')

    @router.get('/checkpoint')
    def checkpoint():
        with session.lock:
            session.save()
            return Response(session.path.read_bytes(),media_type='application/octet-stream',headers={'Content-Disposition':'attachment; filename="ava.pt"'})

    app.include_router(router)
