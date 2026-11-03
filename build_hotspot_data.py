"""Build out/hotspot_data.json keyed by (road name, region), with police-name aliases to official SDI roads."""
import json, re, pandas as pd, numpy as np
REG={'abu_dhabi':'Abu Dhabi','al_ain':'Al Ain','al_dhafra':'Al Dhafra'}
j=json.load(open('data/sdi_roads_match.json',encoding='utf-8'))
match=pd.DataFrame(j['match']).dropna(subset=['road'])[['street','road']]
hq=json.load(open('data/sdi_hq.json',encoding='utf-8')); ex=json.load(open('data/sdi_extra.json',encoding='utf-8'))
rg=json.load(open('data/sdi_roads_region.json',encoding='utf-8')); ex2=json.load(open('data/sdi_extra2.json',encoding='utf-8')); ex3=json.load(open('data/wx_and_extra3.json',encoding='utf-8'))['extra']
stats=pd.concat([pd.DataFrame(rg['stats']),pd.DataFrame(ex['stats']),pd.DataFrame(ex2['stats']),pd.DataFrame(ex3['stats'])]).drop_duplicates(['NAME_AR','Region']).rename(columns={'NAME_AR':'road'})
stats['cls']=stats.rd_class.map({int(k):v for k,v in j['rd_class'].items()})
geo={**hq['geo'],**ex['geo'],**ex2['geo'],**ex3['geo']}
EXACT={(m[0],m[1]):m[2] for m in ex2['map']}   # (police street, region) -> SDI name, from full-name match
# ---- aliases: police street (regex) -> official SDI NAME_AR; region comes from the police city ----
ALIAS=[
 (r'^طريق أبوظبي - العين$|^طريق الشاحنات\(أبوظبي-العين\)$','طريق العين'),            # E22 (truck road police label also used for E22)
 (r'^(شارع|طريق) الشاحنات( القديم| الجديد)?$|^طريق الروضة الريف$','طريق الروضة'),   # E30 Abu Dhabi–Al Ain truck road (Al Rawdah Rd)
 (r'الشارع العام \(أبوظبي - السلع\)|^طريق الشارع العام - |^شارع الرويس - السلع$|^شارع جبل الظنة$|^طريق طريف$|^شارع طريف - الرويس$','طريق الشيخ خليفة بن زايد الدولي'),  # E11
 (r'^طريق أم النار / الشهامة$|^طريق الشيخ زايد$|^شارع الشيخ زايد$|^شارع الشيخ زايد - نفق|^نفق الشيخ زايد|^جسر الشيخ زايد|^جسر السعادة - شارع الشيخ زايد$|^شارع السلام( \(8\))?$|^IP11A شارع السلام','شارع الشيخ زايد بن سلطان'),  # E10 Salam St – Sheikh Zayed Bridge – Shahama
 (r'^شارع العين - دبي$','طريق طحنون بن محمد آل نهيان'),
 (r'^شارع المفرق الشوامخ$|^باتجاه الشوامخ$|^الشوامخ$','شارع الشوامخ'),
 (r'^جزيرة الريم$','شارع الريم'),                                 # E66
 (r'^طريق طريف$|^شارع الأقمار - طريف$|^شارع طريف - بدع زايد$|^شارع ليوا$|^شارع بدع زايد - ليوا$|^شارع مدينة زايد \(الجديد\)$|^شارع بدع زايد - مدينة محمد$','طريق الشيخة سلامة بنت بطي'),  # E45 Tarif–Madinat Zayed–Liwa
 (r'^شارع حبشان - بوحصا$','طريق حبشان'),
 (r'^شارع حميم$|^شارع المقاطرة - حميم$','طريق حميم'),
 (r'^شارع أم الزمول$','طريق أم الزمول'),
 (r'^شارع غياثي|^شارع عرادة - غياثي$|^شارع مزيرعة - عرادة$','شارع غياثي'),              # E15
 (r'^شارع العين - الوجن - القوع$','طريق القوع'),
 (r'^شارع ناهل','طريق ناهل'),
 (r'^شارع الساد - سويحان$|^شارع سويحان - الهير$|^طريق سويحان$','طريق سويحان'),         # E20
 (r'^شارع الشيخ خليفة بن زايد$|^طريق ميناء الشيخ خليفه','شارع الشيخ خليفة بن زايد آل نهيان'),  # E12
]
def alias(street):
    for pat,name in ALIAS:
        if re.search(pat,street): return name
    return None
df=pd.read_parquet('out/accidents_clean.parquet'); inj=df[(df.severity=='injury')&~df.is_duplicate].copy()
inj['n_injured']=pd.to_numeric(inj.n_injured,errors='coerce').fillna(1); inj['year']=inj.year.astype(int); inj['hour']=pd.to_numeric(inj.hour,errors='coerce')
inj=inj.merge(match,on='street',how='left'); inj['Region']=inj.region.map(REG)
al=inj.street.map(alias); ex_map=inj.apply(lambda r: EXACT.get((r.street,r.Region)),axis=1)
inj['road']=al.where(al.notna(),ex_map.where(ex_map.notna(),inj.road))
m=inj.dropna(subset=['road']).merge(stats,on=['road','Region'],how='left',indicator=True)
have=set(geo.keys()); m=m[(m._merge=='both')&m.apply(lambda r:f"{r.road}|{r.Region}" in have,axis=1)]
print('matched',len(m),'of',len(inj),f'({len(m)/len(inj):.0%}); aliased rows:',int(al.notna().sum()))
def agg(g):
    return dict(accidents=int(len(g)),injured=float(g.n_injured.sum()),multi=float((g.n_injured>=2).mean()),night=float(((g.hour>=22)|(g.hour<=5)).mean()) if g.hour.notna().any() else None,
        top_cause=g.reason.value_counts().index[0],top_type=g.collision_type.value_counts().index[0],**{f'y{y%100}':int((g.year==y).sum()) for y in range(2022,2026)})
roads=[]; per={}
for (r,R),g in m.groupby(['road','Region']):
    key=f'{r}|{R}'; s=stats[(stats.road==r)&(stats.Region==R)].iloc[0]
    d=dict(key=key,name=r,region=R,**agg(g),cls=s.cls,lanes=float(s.lanes),speed=float(s.speed),len_m=float(s.len_m)); d['km']=round(d['len_m']/2000,1)
    d['per_km_yr']=round(d['accidents']/d['km']/4,2) if d['km']>=5 else None
    d['police_names']=g.street.value_counts().head(4).index.tolist()
    roads.append(d)
    per[key]={str(y):dict(a=int(len(gy)),i=int(gy.n_injured.sum()),m=round(float((gy.n_injured>=2).mean()),3),n=round(float(((gy.hour>=22)|(gy.hour<=5)).mean()),3) if gy.hour.notna().any() else None,c=gy.reason.value_counts().index[0],t=gy.collision_type.value_counts().index[0]) for y,gy in g.groupby('year')}
keys={r['key'] for r in roads}
# extend corridor roads: if the same SDI name exists in a neighbouring region with no police entries, draw that stretch too
ext_geo={}
for r in roads:
    others=[k for k in geo if k.split('|')[0]==r['name'] and k!=r['key'] and k not in keys]
    def touches(a,b):
        pa=[pt for seg in a for pt in seg[::3]]; pb=[pt for seg in b for pt in seg[::3]]
        return any(abs(x-u)<0.03 and abs(y-v)<0.03 for x,y in pa[::7] for u,v in pb[::7])
    hw=lambda name,reg: (lambda s: len(s) and int(s.iloc[0].rd_class)<=2)(stats[(stats.road==name)&(stats.Region==reg)])
    others=[k for k in others if hw(r['name'],k.split('|')[1]) and touches(geo[r['key']],geo[k])]
    if others and r['km']>=5 and r['cls'] in ('Freeway','Rural Expressway'):
        g=list(geo[r['key']]); extra_km=0
        for k in others:
            g+=geo[k]; s=stats[(stats.road==r['name'])&(stats.Region==k.split('|')[1])]
            if len(s): extra_km+=float(s.iloc[0].len_m)/2000
        ext_geo[r['key']]=g; r['extended']=[k.split('|')[1] for k in others]; r['km']=round(r['km']+extra_km,1)
        r['per_km_yr']=round(r['accidents']/r['km']/4,2) if r['km']>=5 else None
geo={**geo,**ext_geo}
matched_idx=set(m.index)
unm=inj.loc[~inj.index.isin(m.index)] if False else inj[~inj.set_index(['street','date','time']).index.isin(m.set_index(['street','date','time']).index)]
out=dict(roads=roads,geo={k:v for k,v in geo.items() if k in keys},per_year=per,
    year_totals={str(y):dict(a=int(len(g)),i=int(g.n_injured.sum()),m=round(float((g.n_injured>=2).mean()),3)) for y,g in inj.groupby('year')},
    unmatched=unm.street.value_counts().head(14).to_dict(),total_injury_accidents=int(len(inj)),matched=int(len(m)))
# ---- area markers: police labels that are districts, not roads ----
AREAS=[('مدينة الرياض','Riyadh City',54.642,24.300,'Abu Dhabi'),('طرق المطار الدولي','Zayed International Airport roads',54.651,24.433,'Abu Dhabi'),('جسر مدينة خليفة','Khalifa City bridge',54.560,24.420,'Abu Dhabi'),('ياس مول','Yas Mall',54.607,24.488,'Abu Dhabi'),('جزيرة ساس النخل','Sas Al Nakhl Island',54.520,24.430,'Abu Dhabi')]
areas=[]
for ar,en,lon,lat,R in AREAS:
    g=inj[inj.street==ar]
    if not len(g): continue
    areas.append(dict(ar=ar,en=en,lon=lon,lat=lat,region=R,accidents=int(len(g)),injured=float(g.n_injured.sum()),multi=float((g.n_injured>=2).mean()),top_cause=g.reason.value_counts().index[0],per_year={str(y):int(n) for y,n in g.groupby('year').size().items()}))
out['areas']=areas
# ---- what changed: causes and collision types by year (injury accidents, all streets) ----
cy=inj.pivot_table(index='reason',columns='year',values='n_injured',aggfunc='size').fillna(0).astype(int)
cy=cy[cy.sum(axis=1)>=40]
out['cause_years']=[dict(cause=c,**{str(y):int(v) for y,v in row.items()}) for c,row in cy.iterrows()]
ty=inj.pivot_table(index='collision_type',columns='year',values='n_injured',aggfunc='size').fillna(0).astype(int)
out['type_years']=[dict(type=c,**{str(y):int(v) for y,v in row.items()}) for c,row in ty.iterrows()]
# where the red-light rise is: roads by cause 2022 vs 2025 (matched only)
rl=m[m.reason.str.contains('الحمراء')].groupby(['road','Region','year']).size().unstack(fill_value=0)
rl=rl.reindex(columns=[2022,2023,2024,2025],fill_value=0); rl['d']=rl[2025]-rl[2022]
out['redlight_roads']=[dict(key=f"{r}|{R}",name=r,region=R,**{str(y):int(row[y]) for y in [2022,2023,2024,2025]}) for (r,R),row in rl.sort_values('d',ascending=False).head(12).iterrows()]
# ---- residuals: expected 2025 per road from 2022-24 trend of the emirate, vs actual ----
tot={y:int((inj.year==y).sum()) for y in range(2022,2026)}
growth=tot[2025]/((tot[2022]+tot[2023]+tot[2024])/3)
for r in roads:
    base=(r['y22']+r['y23']+r['y24'])/3; exp=base*growth
    r['expected25']=round(exp,1); r['resid25']=round(r['y25']-exp,1); r['resid_pct']=round(100*(r['y25']-exp)/exp,0) if exp>=3 else None
out['growth_factor']=round(growth,3)
json.dump(out,open('out/hotspot_data.json','w',encoding='utf-8'),ensure_ascii=False)
print(len(roads),'road×region entries; top:'); print(pd.DataFrame(roads).sort_values('accidents',ascending=False)[['name','region','accidents','km']].head(12).to_string(index=False))
print('unmatched top:',list(out['unmatched'].items())[:8])
