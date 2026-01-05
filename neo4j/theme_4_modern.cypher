// theme_4_modern.cypher - 主题：现代摩天楼/地标 (商业区)
MERGE (t:Theme {name:'现代摩天楼/地标'})
WITH t
UNWIND [
  {name:'陆家嘴'}, {name:'东方明珠'}, {name:'世纪大道'}, {name:'人民广场'}, {name:'浦东'}, {name:'外滩'}, {name:'黄浦江'}
] AS lm
MATCH (l:Landmark {name:lm.name})
MERGE (t)-[:包含]->(l)
WITH t,l
MERGE (g:Grade {name:l.grade})
MERGE (l)-[:情感评级 {score:l.score}]->(g)
RETURN 'Theme 4 done' AS status;