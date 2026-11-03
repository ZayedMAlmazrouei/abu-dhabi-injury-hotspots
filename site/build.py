import json,csv
U='../'
t=open('template.html',encoding='utf-8').read()
d=open(U+'out/hotspot_data.json',encoding='utf-8').read()
hq=json.load(open(U+'data/sdi_context_hq.json')); bd=json.load(open(U+'data/uae_boundary.json'))
ctx={'land':bd['land'],'borders':bd['borders'],'roads':hq['roads'],'uae':hq['uae']}
fc=[{'region':r['region'],'date':r['date'],'y':float(r['y']),'pred':round(float(r['pred']),1)} for r in csv.DictReader(open(U+'out/forecast_2025.csv'))]
i18n=json.load(open('i18n.json',encoding='utf-8'))
i18n['places']=[[{'en':p[0],'ar':p[1]},p[2],p[3],p[4]] for p in i18n['places']]
html=t.replace('__DATA__',d.replace('</','<\\/')).replace('__CTX__',json.dumps(ctx,separators=(',',':'))).replace('__FC__',json.dumps(fc)).replace('__I18N__',json.dumps(i18n,ensure_ascii=False).replace('</','<\\/'))
open('hotspots.html','w',encoding='utf-8').write(html); print(len(html)//1024,'KB')
