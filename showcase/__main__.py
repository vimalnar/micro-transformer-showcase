import argparse
import json
from pathlib import Path
from .core import Session

parser = argparse.ArgumentParser(description='Local Python-backed Micro-Transformer Showcase: Ava, Chatbot and Device messaging')
parser.add_argument('--showcase', choices=('chatbot','devices'),default='chatbot')
parser.add_argument('--state', type=Path, help='Separate model/dataset state directory')
commands = parser.add_subparsers(dest='command',required=True)
serve = commands.add_parser('serve'); serve.add_argument('--port',type=int,default=8765); serve.add_argument('--device-state',type=Path); serve.add_argument('--ava-state',type=Path)
predict = commands.add_parser('predict'); predict.add_argument('prompt')
train = commands.add_parser('train'); train.add_argument('--examples',type=Path); train.add_argument('--rounds',type=int,default=2); train.add_argument('--rate',type=float,default=.0001)
commands.add_parser('info')
event = commands.add_parser('event'); event.add_argument('place'); event.add_argument('--withheld',action='store_true')
commands.add_parser('interpret'); commands.add_parser('clear-stream')
args = parser.parse_args()
if args.command == 'serve':
    import uvicorn
    from .server import create_app
    uvicorn.run(create_app(args.state,args.device_state,args.ava_state),host='127.0.0.1',port=args.port)
else:
    session = Session(args.state,scope=args.showcase)
    if args.command in ('event','interpret','clear-stream'):
        if args.showcase != 'devices':
            parser.error('Simulation commands require --showcase devices')
        from .devices import Simulation
        simulation = Simulation(session)
        result = simulation.send(args.place,not args.withheld) if args.command == 'event' else (simulation.interpret() if args.command == 'interpret' else simulation.clear())
        print(json.dumps(result,indent=2))
    elif args.command == 'predict':
        print(json.dumps(session.predict(args.prompt),indent=2))
    elif args.command == 'train':
        recipe = dict(session.recipe or session.draft)
        if args.examples:
            recipe = dict(text=args.examples.read_text(),rounds=args.rounds,rate=args.rate)
        session.start(recipe,background=False)
        print(json.dumps(session.status(),indent=2))
        if session.error:
            raise SystemExit(1)
    else:
        print(json.dumps(session.status(),indent=2))
