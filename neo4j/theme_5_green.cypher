// theme_5_green.cypher - 主题：公园/绿色空间
MERGE (t:Theme {name:'公园/绿色空间'})
WITH t
UNWIND [
  {name:'人民公园'}, {name:'世纪公园'}, {name:'静安公园'}, {name:'长风'}
] AS lm
MATCH (l:Landmark {name:lm.name})
MERGE (t)-[:包含]->(l)
WITH t,l
MERGE (g:Grade {name:l.grade})
MERGE (l)-[:情感评级 {score:l.score}]->(g)
RETURN 'Theme 5 done' AS status;