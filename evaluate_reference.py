"""Evaluation only: this file is never imported by the automatic cutter."""
import argparse,json
from pathlib import Path


def evaluate(report,references):
    predictions=[r for r in report['segments'] if r['status']=='keep']
    pairs=[]
    for i,(a,b) in enumerate(references):
        for j,r in enumerate(predictions):
            overlap=max(0,min(b,r['end'])-max(a,r['release']))
            union=max(b,r['end'])-min(a,r['release'])
            pairs.append((overlap/union if union else 0,i,j))
    used_ref=set();used_pred=set();matches=[]
    for score,i,j in sorted(pairs,reverse=True):
        if score<.5 or i in used_ref or j in used_pred:continue
        used_ref.add(i);used_pred.add(j)
        a,b=references[i];r=predictions[j]
        matches.append(dict(reference=[a,b],predicted_release=r['release'],predicted_end=r['end'],
            export_interval=[r['start'],r['end']],duration=r['duration'],
            start_error=r['release']-a,end_error=r['end']-b,iou=score))
    return dict(evaluation_kind='development_video_approximate_annotations_not_independent_validation',
                matched=len(matches),reference_count=len(references),candidate_count=len(predictions),
                missed=[r for i,r in enumerate(references) if i not in used_ref],
                unmatched_candidates=[p for j,p in enumerate(predictions) if j not in used_pred],
                matches=sorted(matches,key=lambda m:m['reference'][0]),
                last_analyzed=report['last_analyzed'],sampled_frames=report['sampled_frames'],
                dry_run=report['dry_run'],accuracy_validated=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('report',type=Path)
    parser.add_argument('--references',type=Path,default=Path(__file__).parent/'development'/'reference_intervals.json')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'development'/'validation.json')
    args=parser.parse_args()
    result=evaluate(json.loads(args.report.read_text(encoding='utf-8')),
                    json.loads(args.references.read_text(encoding='utf-8'))['intervals'])
    result['report']=str(args.report.resolve())
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
