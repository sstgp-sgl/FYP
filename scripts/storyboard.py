#!/usr/bin/env python3
"""Scene demos, deterministic compilation, bounded LLM understanding and independent validation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    demo = subs.add_parser('demo')
    demo.add_argument('--config', default='configs/storyboard.yaml')
    demo.add_argument('--out', required=True)
    demo.add_argument('--red-ply', help='Optional synthetic test fixture override; never a real-asset import')
    validate = subs.add_parser('validate')
    validate.add_argument('--run', required=True)
    importer = subs.add_parser('import-asset')
    importer.add_argument('--ply', required=True)
    importer.add_argument('--metadata', required=True, help='JSON with source_image, producer, name, optional up/forward/yaw/height')
    importer.add_argument('--library', required=True)
    compose = subs.add_parser('compose-assets')
    compose.add_argument('--library', required=True)
    compose.add_argument('--instances', required=True, help='JSON list of exactly two asset_id/instance_id objects, first goes left')
    compose.add_argument('--out', required=True)
    compose.add_argument('--allow-provisional', action='store_true')
    layout = subs.add_parser('layout')
    layout.add_argument('--asset-root', required=True)
    layout.add_argument('--config',default='configs/storyboard.yaml')
    layout.add_argument('--scene', required=True)
    layout.add_argument('--request', required=True)
    layout.add_argument('--out', required=True)
    layout.add_argument('--allow-provisional', action='store_true')
    compile_parser=subs.add_parser('compile')
    compile_parser.add_argument('--plan',required=True)
    compile_parser.add_argument('--asset-root',required=True)
    compile_parser.add_argument('--scene',required=True)
    compile_parser.add_argument('--out',required=True)
    compile_parser.add_argument('--config',default='configs/storyboard.yaml')
    compile_parser.add_argument('--allow-provisional',action='store_true')
    understanding=subs.add_parser('understand')
    understanding.add_argument('--story-file',required=True)
    understanding.add_argument('--catalog',required=True,help='Confirmed instance names JSON')
    understanding.add_argument('--asset-root',required=True)
    understanding.add_argument('--scene',required=True)
    understanding.add_argument('--out',required=True)
    understanding.add_argument('--config',default='configs/storyboard.yaml')
    understanding.add_argument('--allow-provisional',action='store_true',help='Allow unconfirmed source orientation, never unconfirmed names')
    providers=understanding.add_mutually_exclusive_group(required=True)
    providers.add_argument('--api-model',help='DeepSeek official model ID; key is read from --api-key-env')
    understanding.add_argument('--api-key-env',default='FYP_LLM_API_KEY')
    providers.add_argument('--model-path',help='Pinned local Transformers model; CUDA Slurm allocation required')
    providers.add_argument('--fixture-responses',help='JSON list of scripted raw responses; labelled synthetic/test')
    providers.add_argument('--replay-attempt',help='Saved raw response; labelled replay, never a new real model call')
    understanding.add_argument('--max-calls',type=int,default=3)
    understanding.add_argument('--call-timeout',type=float,default=180)
    understanding.add_argument('--max-output-tokens',type=int,default=3072)
    args = parser.parse_args()
    from src.storyboard import service
    try:
        if args.command == 'understand':
            from src.storyboard.llm_provider import TransformersProvider,FixtureProvider,ReplayProvider,DeepSeekProvider
            from src.storyboard.llm_service import understand
            provider=(DeepSeekProvider(args.api_model,args.api_key_env) if args.api_model else TransformersProvider(args.model_path) if args.model_path else ReplayProvider(args.replay_attempt) if args.replay_attempt else FixtureProvider(json.loads(Path(args.fixture_responses).read_text())))
            report=understand(Path(args.story_file).read_text(),json.loads(Path(args.catalog).read_text()),args.asset_root,args.scene,args.out,provider,limits=dict(max_calls=args.max_calls,call_timeout_seconds=args.call_timeout,max_output_tokens=args.max_output_tokens),config_path=args.config,allow_provisional=args.allow_provisional)
            print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
            return report['exit_code']
        if args.command == 'validate' and (Path(args.run)/'run.json').is_file() and json.loads((Path(args.run)/'run.json').read_text()).get('pipeline') == 'story_understanding_p5':
            from src.storyboard.llm_service import validate as validate_understanding
            report=validate_understanding(args.run)
            print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
            return 0 if report['passed'] else 1
        if args.command == 'compile':
            from src.storyboard.story_service import compile_story
            report=compile_story(args.plan,args.asset_root,args.scene,args.out,args.config,args.allow_provisional)
            print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
            return 0 if report['passed'] else 1
        if args.command == 'validate' and (Path(args.run)/'run.json').is_file() and json.loads((Path(args.run)/'run.json').read_text()).get('pipeline') == 'storyboard_p4':
            from src.storyboard.story_service import validate as validate_story
            report=validate_story(args.run)
            print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
            return 0 if report['passed'] else 1
        if args.command == 'layout':
            from src.storyboard.spatial_service import compile_layout
            report=compile_layout(args.asset_root,args.scene,args.request,args.out,args.allow_provisional,config_path=args.config)
            print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
            return 0 if report['passed'] else 1
        if args.command == 'validate' and (Path(args.run)/'run.json').is_file() and json.loads((Path(args.run)/'run.json').read_text()).get('pipeline') == 'spatial_layout_p3':
            from src.storyboard.spatial_service import validate as validate_spatial
            report=validate_spatial(args.run)
            print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
            return 0 if report['passed'] else 1
        if args.command == 'import-asset':
            from src.storyboard.assets import import_asset
            report = import_asset(args.library, args.ply, json.loads(Path(args.metadata).read_text()))
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if args.command == 'compose-assets':
            from src.storyboard.real_scene import compose
            report = compose(args.library, json.loads(Path(args.instances).read_text()), args.out, args.allow_provisional)
        elif args.command == 'validate' and (Path(args.run)/'run.json').is_file() and json.loads((Path(args.run)/'run.json').read_text()).get('source') == 'real':
            from src.storyboard.real_scene import validate as validate_real
            report = validate_real(args.run)
        else:
            report = (service.demo(args.config, args.out, args.red_ply) if args.command == 'demo' else service.validate(args.run))
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if report['passed'] else 1
    except Exception as exc:
        print(json.dumps(dict(status='failed', stage=args.command, error_type=type(exc).__name__, error=str(exc)), ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
