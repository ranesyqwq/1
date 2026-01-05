// theme_3_streets.cypher - 主题：特色街巷/酒吧街
MERGE (t:Theme {name:'特色街巷/酒吧街'})
WITH t
UNWIND [
  {name:'武康路'}, {name:'安福路'}, {name:'衡山路'}, {name:'长乐路'}, {name:'永康路'}, {name:'甜爱路'}, {name:'复兴中路'}, {name:'多伦路'}, {name:'永康路'}
] AS lm
MATCH (l:Landmark {name:lm.name})
MERGE (t)-[:包含]->(l)
WITH t,l
MERGE (g:Grade {name:l.grade})
MERGE (l)-[:情感评级 {score:l.score}]->(g)
RETURN 'Theme 3 done' AS status;