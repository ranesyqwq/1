// theme_7_food.cypher - 主题：美食/小吃
MERGE (t:Theme {name:'美食/小吃'})
WITH t
UNWIND [
  {name:'城隍庙'}, {name:'福州路'}, {name:'七宝'}, {name:'百乐门'}
] AS lm
MATCH (l:Landmark {name:lm.name})
MERGE (t)-[:包含]->(l)
WITH t,l
MERGE (g:Grade {name:l.grade})
MERGE (l)-[:情感评级 {score:l.score}]->(g)
RETURN 'Theme 7 done' AS status;