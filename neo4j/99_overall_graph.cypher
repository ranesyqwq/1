// 99_overall_graph.cypher - 包含所有主题与其打卡点的主知识图谱
// 先执行 00_create_landmarks.cypher 创建人地节点

// 定义主题与其打卡点（简单合并所有主题清单）
UNWIND [
  {theme:'文化场馆/历史街区', landmarks:['豫园','外白渡桥','城隍庙','田子坊','静安寺','龙华寺','朱家角']},
  {theme:'购物/商业街区', landmarks:['南京路','淮海中路','淮海路','陕西南路','徐家汇','北京路']},
  {theme:'特色街巷/酒吧街', landmarks:['武康路','安福路','衡山路','长乐路','永康路','甜爱路','复兴中路','多伦路']},
  {theme:'现代摩天楼/地标', landmarks:['陆家嘴','东方明珠','世纪大道','人民广场','浦东','外滩','黄浦江']},
  {theme:'公园/绿色空间', landmarks:['人民公园','世纪公园','静安公园','长风']},
  {theme:'艺术区/创意园', landmarks:['M50','愚园路','共青团','威海路']},
  {theme:'美食/小吃', landmarks:['城隍庙','福州路','七宝','百乐门']}
] AS row
MERGE (t:Theme {name:row.theme})
WITH t,row
UNWIND row.landmarks AS lmName
MATCH (l:Landmark {name:lmName})
MERGE (t)-[:包含]->(l)
WITH t,l
MERGE (g:Grade {name:l.grade})
MERGE (l)-[:情感评级 {score:l.score}]->(g)
RETURN 'Overall graph created' AS status;