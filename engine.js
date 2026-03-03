// ── Theme ─────────────────────────────────────────────────────────────────────
(function(){
  var saved=localStorage.getItem('elisa-theme');
  if(saved==='light')document.body.classList.add('light-mode');
  updateThemeIcon();
})();

function toggleTheme(){
  var isLight=document.body.classList.toggle('light-mode');
  localStorage.setItem('elisa-theme',isLight?'light':'dark');
  updateThemeIcon();
}

function updateThemeIcon(){
  var icon=document.getElementById('themeIcon');
  if(!icon)return;
  var isLight=document.body.classList.contains('light-mode');
  icon.innerHTML=isLight
    ? '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>'
    : '<path d="M21 12.79A9 9 0 1111.21 3a7 7 0 009.79 9.79z"/>';
}

// ── Welcome Modal ─────────────────────────────────────────────────────────────
function closeModal(){
  var m=document.getElementById('welcomeModal');
  if(!m)return;
  m.classList.add('closing');
  setTimeout(function(){m.remove();},320);
}

document.addEventListener('keydown',function(e){
  if(e.key==='Escape')closeModal();
});

// ── State ────────────────────────────────────────────────────────────────────
var S={refConc:[],refAbs:[],model:'linear',fitFn:null,inverseFn:null,r2:null,curveReady:false,sampleResults:[]};

// ── Particles ─────────────────────────────────────────────────────────────────
(function(){var c=document.getElementById('particles'),cols=['#00e5ff','#7b61ff','#ff4d8d'];
for(var i=0;i<22;i++){var p=document.createElement('div');p.className='particle';
p.style.cssText='left:'+Math.random()*100+'vw;animation-duration:'+(9+Math.random()*14)+'s;animation-delay:'+Math.random()*16+'s;background:'+cols[i%3]+';opacity:'+(0.3+Math.random()*0.6);
c.appendChild(p);}})();

// ── Helpers ───────────────────────────────────────────────────────────────────
function switchTab(n){
  if(n===2&&!S.curveReady)return;
  document.querySelectorAll('.tab').forEach(function(t,i){t.classList.toggle('active',i===n-1);});
  document.querySelectorAll('.panel').forEach(function(p,i){p.classList.toggle('active',i===n-1);});
}

function selectModel(m){
  S.model=m;
  document.querySelectorAll('.model-card').forEach(function(c){c.classList.remove('active');});
  document.getElementById('m-'+m).classList.add('active');
}

function parseVals(txt){
  return txt.split('\n').map(function(l){return l.trim().replace(',','.');})
    .filter(function(l){return l!=='';}).map(parseFloat).filter(function(v){return !isNaN(v);});
}

function parseTwoCols(txt){
  var conc=[],abs=[];
  txt.split('\n').filter(function(l){return l.trim();}).forEach(function(l){
    var p=l.trim().split(/\t|,|;/).map(function(x){return parseFloat(x.replace(',','.'));});
    if(p.length>=2&&!isNaN(p[0])&&!isNaN(p[1])){conc.push(p[0]);abs.push(p[1]);}
  });
  return{conc:conc,abs:abs};
}

function linspace(a,b,n){var r=[];for(var i=0;i<n;i++)r.push(a+(b-a)*i/(n-1));return r;}

function mean(a){return a.reduce(function(s,v){return s+v;},0)/a.length;}

function calcR2(actual,pred){
  var m=mean(actual);
  var sst=actual.reduce(function(s,v){return s+(v-m)*(v-m);},0);
  var ssr=actual.reduce(function(s,v,i){return s+(v-pred[i])*(v-pred[i]);},0);
  return 1-ssr/sst;
}

function calcRMSE(actual,pred){
  var mse=actual.reduce(function(s,v,i){return s+(v-pred[i])*(v-pred[i]);},0)/actual.length;
  return Math.sqrt(mse);
}

// ── Regression ────────────────────────────────────────────────────────────────
function fitLinear(x,y){
  var n=x.length,sx=0,sy=0,sxy=0,sx2=0;
  for(var i=0;i<n;i++){sx+=x[i];sy+=y[i];sxy+=x[i]*y[i];sx2+=x[i]*x[i];}
  var a=(n*sxy-sx*sy)/(n*sx2-sx*sx),b=(sy-a*sx)/n;
  return{fn:function(xv){return a*xv+b;},inv:function(yv){return(yv-b)/a;},label:'y = '+a.toFixed(4)+'x + '+b.toFixed(4)};
}

function fitPoly2(x,y){
  var n=x.length,XtX=[[0,0,0],[0,0,0],[0,0,0]],Xty=[0,0,0];
  for(var i=0;i<n;i++){
    var row=[1,x[i],x[i]*x[i]];
    for(var j=0;j<3;j++){Xty[j]+=row[j]*y[i];for(var k=0;k<3;k++)XtX[j][k]+=row[j]*row[k];}
  }

  var c=solve3(XtX,Xty);

  return{
    fn:function(xv){return c[0]+c[1]*xv+c[2]*xv*xv;},
    inv:function(yv){var disc=c[1]*c[1]-4*c[2]*(c[0]-yv);if(disc<0)return NaN;var r1=(-c[1]+Math.sqrt(disc))/(2*c[2]),r2=(-c[1]-Math.sqrt(disc))/(2*c[2]);return r1>0?r1:r2;},
    label:'y = '+c[2].toFixed(5)+'x² + '+c[1].toFixed(4)+'x + '+c[0].toFixed(4)
  };
}

function solve3(A,b){
  var M=A.map(function(r,i){return r.slice().concat(b[i]);});
  for(var col=0;col<3;col++){
    var mx=col;for(var row=col+1;row<3;row++)if(Math.abs(M[row][col])>Math.abs(M[mx][col]))mx=row;
    var tmp=M[col];M[col]=M[mx];M[mx]=tmp;
    for(var row=col+1;row<3;row++){var f=M[row][col]/M[col][col];for(var k=col;k<=3;k++)M[row][k]-=f*M[col][k];}
  }

  var x=[0,0,0];
  for(var i=2;i>=0;i--){x[i]=M[i][3];for(var j=i+1;j<3;j++)x[i]-=M[i][j]*x[j];x[i]/=M[i][i];}
  return x;
}

function fitExponential(x,y){
  var lny=y.map(function(v){return Math.log(Math.max(v,1e-10));});
  var r=fitLinear(x,lny),a=Math.exp(r.inv(0)/(-r.fn(0)/r.fn(1)));
  // simpler: ln(y)=ln(a)+bx
  var n=x.length,sx=0,slny=0,sxlny=0,sx2=0;
  for(var i=0;i<n;i++){sx+=x[i];slny+=lny[i];sxlny+=x[i]*lny[i];sx2+=x[i]*x[i];}
  var b=(n*sxlny-sx*slny)/(n*sx2-sx*sx),lna=(slny-b*sx)/n;
  a=Math.exp(lna);
  return{fn:function(xv){return a*Math.exp(b*xv);},inv:function(yv){return Math.log(yv/a)/b;},label:'y = '+a.toFixed(4)+'·e^('+b.toFixed(4)+'x)'};
}

function fitLog(x,y){
  var lnx=x.map(function(v){return Math.log(Math.max(v,1e-10));});
  var r=fitLinear(lnx,y),a=r.fn(1)-r.fn(0),b=r.fn(0);
  return{fn:function(xv){return a*Math.log(Math.max(xv,1e-10))+b;},inv:function(yv){return Math.exp((yv-b)/a);},label:'y = '+a.toFixed(4)+'·ln(x) + '+b.toFixed(4)};
}

function fitPower(x,y){
  var lnx=x.map(function(v){return Math.log(Math.max(v,1e-10));});
  var lny=y.map(function(v){return Math.log(Math.max(v,1e-10));});
  var r=fitLinear(lnx,lny),b=r.fn(1)-r.fn(0),lna=r.fn(0);
  var a=Math.exp(lna);
  return{fn:function(xv){return a*Math.pow(Math.max(xv,1e-10),b);},inv:function(yv){return Math.pow(yv/a,1/b);},label:'y = '+a.toFixed(4)+'·x^'+b.toFixed(4)};
}

function nlsFit(p0,x,y,mFn,maxIter){
  var p=p0.slice(),lr=0.001,prev=Infinity;
  for(var iter=0;iter<maxIter;iter++){
    var pred=x.map(function(xv){return mFn(p,xv);});
    var loss=pred.reduce(function(s,v,i){return s+(v-y[i])*(v-y[i]);},0);
    if(Math.abs(prev-loss)<1e-13)break;prev=loss;
    var grad=p.map(function(_,j){
      var d=Math.abs(p[j])*1e-5+1e-8,pp=p.slice();pp[j]+=d;
      var pred2=x.map(function(xv){return mFn(pp,xv);});
      var l2=pred2.reduce(function(s,v,i){return s+(v-y[i])*(v-y[i]);},0);
      return(l2-loss)/d;
    });
    var gnorm=Math.sqrt(grad.reduce(function(s,v){return s+v*v;},0));
    if(gnorm<1e-12)break;
    var step=lr/gnorm;
    p=p.map(function(v,j){return v-step*grad[j];});
    if(iter%300===0)lr*=0.92;
  }
  return p;
}

function invertNum(fn,yv,xRef){
  var lo=Math.min.apply(null,xRef)*0.001,hi=Math.max.apply(null,xRef)*1000;
  for(var i=0;i<80;i++){var mid=(lo+hi)/2;if((fn(lo)-yv)*(fn(mid)-yv)<=0)hi=mid;else lo=mid;}
  return(lo+hi)/2;
}

function fit4PL(x,y){
  var A=Math.min.apply(null,y)*0.9,D=Math.max.apply(null,y)*1.1,C=mean(x);
  var mFn=function(p,xv){return p[3]+(p[0]-p[3])/(1+Math.pow(Math.max(xv/p[2],1e-10),p[1]));};
  var p=nlsFit([A,1.5,C,D],x,y,mFn,4000);
  return{
    fn:function(xv){return p[3]+(p[0]-p[3])/(1+Math.pow(Math.max(xv/p[2],1e-10),p[1]));},
    inv:function(yv){var r=(p[0]-p[3])/(yv-p[3])-1;if(r<=0)return NaN;return p[2]*Math.pow(r,1/p[1]);},
    label:'4PL: A='+p[0].toFixed(3)+', B='+p[1].toFixed(3)+', C='+p[2].toFixed(3)+', D='+p[3].toFixed(3)
  };
}

function fit5PL(x,y){
  var A=Math.min.apply(null,y)*0.9,D=Math.max.apply(null,y)*1.1,C=mean(x);
  var mFn=function(p,xv){return p[3]+(p[0]-p[3])/Math.pow(1+Math.pow(Math.max(xv/p[2],1e-10),p[1]),p[4]);};
  var p=nlsFit([A,1.5,C,D,1.0],x,y,mFn,5000);
  var fn=function(xv){return p[3]+(p[0]-p[3])/Math.pow(1+Math.pow(Math.max(xv/p[2],1e-10),p[1]),p[4]);};
  return{fn:fn,inv:function(yv){return invertNum(fn,yv,x);},
    label:'5PL: A='+p[0].toFixed(3)+', B='+p[1].toFixed(3)+', C='+p[2].toFixed(3)+', D='+p[3].toFixed(3)+', E='+p[4].toFixed(3)};
}

function fitSpline(x,y){
  var sorted=x.map(function(v,i){return[v,y[i]];}).sort(function(a,b){return a[0]-b[0];});
  var xs=sorted.map(function(v){return v[0];}),ys=sorted.map(function(v){return v[1];}),n=xs.length;
  var h=xs.slice(1).map(function(v,i){return v-xs[i];});
  var alpha=new Array(n).fill(0);
  for(var i=1;i<n-1;i++)alpha[i]=(3/h[i])*(ys[i+1]-ys[i])-(3/h[i-1])*(ys[i]-ys[i-1]);
  var l=new Array(n).fill(1),mu=new Array(n).fill(0),z=new Array(n).fill(0);
  for(var i=1;i<n-1;i++){l[i]=2*(xs[i+1]-xs[i-1])-h[i-1]*mu[i-1];mu[i]=h[i]/l[i];z[i]=(alpha[i]-h[i-1]*z[i-1])/l[i];}
  var c=new Array(n).fill(0),b=new Array(n-1).fill(0),d=new Array(n-1).fill(0);
  for(var j=n-2;j>=0;j--){c[j]=z[j]-mu[j]*c[j+1];b[j]=(ys[j+1]-ys[j])/h[j]-h[j]*(c[j+1]+2*c[j])/3;d[j]=(c[j+1]-c[j])/(3*h[j]);}
  var spFn=function(xv){
    var i=xs.findIndex(function(v){return v>xv;})-1;
    if(i<0)i=0;if(i>=n-1)i=n-2;
    var dx=xv-xs[i];return ys[i]+b[i]*dx+c[i]*dx*dx+d[i]*dx*dx*dx;
  };
  return{fn:spFn,inv:function(yv){return invertNum(spFn,yv,xs);},label:'Cubic Spline ('+n+' knots)'};
}

// ── Fit ───────────────────────────────────────────────────────────────────────
var refChartI=null,residChartI=null,sampleChartI=null;

function fitCurve(){
  var conc,abs;
  var tc=document.getElementById('twoColInput').value.trim();
  if(tc){var p=parseTwoCols(tc);conc=p.conc;abs=p.abs;}
  else{conc=parseVals(document.getElementById('concInput').value);abs=parseVals(document.getElementById('absInput').value);}
  if(conc.length<3||abs.length<3){showN('⚠️ Need at least 3 reference points.');return;}
  if(conc.length!==abs.length){showN('⚠️ Concentration and absorbance must have equal length.');return;}
  S.refConc=conc;S.refAbs=abs;
  var r;
  try{
    switch(S.model){
      case'linear':r=fitLinear(conc,abs);break;
      case'poly2':r=fitPoly2(conc,abs);break;
      case'exponential':r=fitExponential(conc,abs);break;
      case'log':r=fitLog(conc,abs);break;
      case'power':r=fitPower(conc,abs);break;
      case'4pl':r=fit4PL(conc,abs);break;
      case'5pl':r=fit5PL(conc,abs);break;
      case'spline':r=fitSpline(conc,abs);break;
      default:r=fitLinear(conc,abs);
    }
  }catch(e){showN('❌ Fitting failed: '+e.message);return;}
  S.fitFn=r.fn;S.inverseFn=r.inv;
  var pred=conc.map(r.fn);
  S.r2=calcR2(abs,pred);
  var rmse=calcRMSE(abs,pred);

  ['stat-r2','stat-rmse','stat-n','stat-model'].forEach(function(id){
    var el=document.getElementById(id);
    el.classList.remove('updated');
    void el.offsetWidth; // reflow to retrigger animation
    el.classList.add('updated');
  });
  document.getElementById('stat-r2').textContent=S.r2.toFixed(4);
  document.getElementById('stat-rmse').textContent=rmse.toFixed(5);
  document.getElementById('stat-n').textContent=conc.length;
  document.getElementById('stat-model').textContent=S.model.toUpperCase();

  var badge=document.getElementById('r2badge');
  badge.textContent='R² = '+S.r2.toFixed(4);
  badge.className='chart-badge '+(S.r2>=0.99?'r2-good':S.r2>=0.95?'r2-ok':'r2-poor');
  var eq=document.getElementById('curveEquation');eq.style.display='block';eq.textContent='✓ Fitted: '+r.label;
  drawRefChart(conc,abs,r.fn);drawResidChart(conc,abs,pred);
  S.curveReady=true;
  document.getElementById('tab2').classList.remove('disabled');
  document.getElementById('lockedOverlay').style.display='none';
  showN('✅ Curve fitted · R² = '+S.r2.toFixed(4));
}

// ── Charts ────────────────────────────────────────────────────────────────────
Chart.defaults.color='#8899bb';Chart.defaults.borderColor='#1a2540';
function mkOpts(xtitle,ytitle){
  return{responsive:true,animation:{duration:600},plugins:{legend:{labels:{color:'#8899bb',font:{family:'JetBrains Mono',size:11}}},tooltip:{backgroundColor:'#0b1022',borderColor:'#1a2540',borderWidth:1}},scales:{x:{title:{display:true,text:xtitle,color:'#8899bb'},grid:{color:'#1a2540'},ticks:{color:'#8899bb'}},y:{title:{display:true,text:ytitle,color:'#8899bb'},grid:{color:'#1a2540'},ticks:{color:'#8899bb'}}}};
}

function drawRefChart(conc,abs,fn){
  if(refChartI)refChartI.destroy();
  var xmin=Math.min.apply(null,conc),xmax=Math.max.apply(null,conc);
  var cx=linspace(xmin,xmax,200),cy=cx.map(fn);
  var ctx=document.getElementById('refChart').getContext('2d');
  refChartI=new Chart(ctx,{type:'scatter',data:{datasets:[
    {label:'Reference Points',data:conc.map(function(v,i){return{x:v,y:abs[i]};}),backgroundColor:'rgba(0,229,255,0.35)',borderColor:'#00e5ff',pointRadius:6,pointHoverRadius:8,pointBorderWidth:2},
    {label:'Fitted Curve',data:cx.map(function(v,i){return{x:v,y:cy[i]};}),type:'line',borderColor:'#7b61ff',borderWidth:2.5,pointRadius:0,tension:0.4,fill:false}
  ]},options:mkOpts('Concentration','Absorbance (OD)')});
}

function drawResidChart(conc,abs,pred){
  if(residChartI)residChartI.destroy();
  var xmin=Math.min.apply(null,conc),xmax=Math.max.apply(null,conc);
  var ctx=document.getElementById('residChart').getContext('2d');
  residChartI=new Chart(ctx,{type:'scatter',data:{datasets:[
    {label:'Residuals',data:conc.map(function(v,i){return{x:v,y:abs[i]-pred[i]};}),backgroundColor:'#ff4d8d',pointRadius:5},
    {label:'Zero',data:[{x:xmin,y:0},{x:xmax,y:0}],type:'line',borderColor:'rgba(255,255,255,0.2)',borderDash:[4,4],pointRadius:0,fill:false}
  ]},options:mkOpts('Concentration','Residual')});
}

// ── Predict ───────────────────────────────────────────────────────────────────
function predictConcentrations(){
  if(!S.curveReady)return;
  var absVals=parseVals(document.getElementById('sampleAbsInput').value);
  var lbls=document.getElementById('sampleLabels').value.split('\n').map(function(l){return l.trim();}).filter(function(l){return l;});
  if(!absVals.length){showN('⚠️ Enter sample absorbance values.');return;}
  var amin=Math.min.apply(null,S.refAbs),amax=Math.max.apply(null,S.refAbs);
  S.sampleResults=absVals.map(function(ab,i){
    var lbl=lbls[i]||'Sample_'+String(i+1).padStart(2,'0');
    var conc=S.inverseFn(ab);
    return{label:lbl,abs:ab,conc:conc,oor:ab<amin||ab>amax};
  });
  var valid=S.sampleResults.filter(function(r){return!isNaN(r.conc)&&r.conc>0;});
  var oor=S.sampleResults.filter(function(r){return r.oor;}).length;
  document.getElementById('s-stat-n').textContent=S.sampleResults.length;
  document.getElementById('s-stat-min').textContent=valid.length?Math.min.apply(null,valid.map(function(r){return r.conc;})).toFixed(3):'—';
  document.getElementById('s-stat-max').textContent=valid.length?Math.max.apply(null,valid.map(function(r){return r.conc;})).toFixed(3):'—';
  document.getElementById('s-stat-oor').textContent=oor;
  document.getElementById('sampleStats').style.display='grid';
  var tbody=document.getElementById('resultsBody');tbody.innerHTML='';
  S.sampleResults.forEach(function(r,i){
    var tr=document.createElement('tr');
    var cv=isNaN(r.conc)||r.conc<=0?'N/A':r.conc.toFixed(4);
    var fl=r.oor?'<span class="flag">OOR</span>':'<span class="ok-flag">✓ OK</span>';
    tr.innerHTML='<td>'+(i+1)+'</td><td>'+r.label+'</td><td>'+r.abs.toFixed(4)+'</td><td class="conc-val">'+cv+'</td><td>'+fl+'</td>';
    tbody.appendChild(tr);
  });
  document.getElementById('resultsWrap').style.display='block';
  document.getElementById('resultsBadge').textContent=S.sampleResults.length+' samples';
  document.getElementById('copySection').style.display='block';
  drawSampleChart();
  showN('✅ '+S.sampleResults.length+' samples quantified');
}

function drawSampleChart(){
  if(sampleChartI)sampleChartI.destroy();
  var xmin=Math.min.apply(null,S.refConc),xmax=Math.max.apply(null,S.refConc);
  var cx=linspace(xmin*0.01,xmax*1.5,300),cy=cx.map(S.fitFn);
  document.getElementById('sampleChartWrap').style.display='block';
  var ctx=document.getElementById('sampleChart').getContext('2d');
  sampleChartI=new Chart(ctx,{type:'scatter',data:{datasets:[
    {label:'Fitted Curve',data:cx.map(function(v,i){return{x:v,y:cy[i]};}),type:'line',borderColor:'rgba(123,97,255,0.65)',borderWidth:2,pointRadius:0,fill:false},
    {label:'Reference',data:S.refConc.map(function(v,i){return{x:v,y:S.refAbs[i]};}),backgroundColor:'rgba(0,229,255,0.45)',borderColor:'#00e5ff',pointRadius:5},
    {label:'Samples',data:S.sampleResults.map(function(r){return{x:r.conc,y:r.abs};}),backgroundColor:'#ff4d8d',borderColor:'#ff4d8d',pointRadius:7,pointStyle:'triangle'}
  ]},options:mkOpts('Concentration','Absorbance (OD)')});
}

// ── Export ────────────────────────────────────────────────────────────────────
function getRows(){return S.sampleResults.map(function(r,i){return{Index:i+1,Label:r.label,Absorbance:r.abs,Concentration:isNaN(r.conc)||r.conc<=0?'N/A':r.conc,Status:r.oor?'Out of Range':'OK'};});}

function copyTable(){
  var d=getRows(),h=Object.keys(d[0]).join('\t');
  var txt=[h].concat(d.map(function(r){return Object.values(r).join('\t');})).join('\n');
  navigator.clipboard.writeText(txt).then(function(){showN('✅ Table copied — paste directly into Excel');});
}

function copyCSV(){
  var d=getRows(),h=Object.keys(d[0]).join(',');
  var txt=[h].concat(d.map(function(r){return Object.values(r).join(',');})).join('\n');
  navigator.clipboard.writeText(txt).then(function(){showN('✅ CSV copied to clipboard');});
}

function downloadCSV(){var d=getRows(),h=Object.keys(d[0]).join(','),rows=d.map(function(r){return Object.values(r).join(',');});dlFile([h].concat(rows).join('\n'),'elisa_results.csv','text/csv');showN('⬇️ CSV downloaded');}

function downloadTSV(){var d=getRows(),h=Object.keys(d[0]).join('\t'),rows=d.map(function(r){return Object.values(r).join('\t');});dlFile([h].concat(rows).join('\n'),'elisa_results.tsv','text/tab-separated-values');showN('⬇️ TSV downloaded');}

function downloadJSON(){
  var data={model:S.model,r2:S.r2,referenceData:S.refConc.map(function(v,i){return{concentration:v,absorbance:S.refAbs[i]};}),sampleResults:S.sampleResults};
  dlFile(JSON.stringify(data,null,2),'elisa_results.json','application/json');showN('⬇️ JSON downloaded');
}

function downloadXLSX(){
  var wb=XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb,XLSX.utils.json_to_sheet(getRows()),'Results');
  XLSX.utils.book_append_sheet(wb,XLSX.utils.json_to_sheet(S.refConc.map(function(v,i){return{Concentration:v,Absorbance:S.refAbs[i],Predicted:S.fitFn(v)};})),'Reference_Curve');
  XLSX.utils.book_append_sheet(wb,XLSX.utils.json_to_sheet([{Key:'Model',Value:S.model},{Key:'R2',Value:S.r2},{Key:'N_Reference',Value:S.refConc.length},{Key:'N_Samples',Value:S.sampleResults.length}]),'Metadata');
  XLSX.writeFile(wb,'elisa_results.xlsx');showN('⬇️ Excel file downloaded');
}

function dlFile(content,name,type){var b=new Blob([content],{type:type}),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download=name;a.click();URL.revokeObjectURL(u);}

// ── Notification ──────────────────────────────────────────────────────────────
var ntimer;
function showN(msg){var el=document.getElementById('notification');el.textContent=msg;el.classList.add('show');clearTimeout(ntimer);ntimer=setTimeout(function(){el.classList.remove('show');},3800);}
