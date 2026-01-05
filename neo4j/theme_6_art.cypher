// theme_6_art.cypher - 主题：艺术区/创意园
MERGE (t:Theme {name:'艺术区/创意园'})
WITH t
UNWIND [
  {name:'M50'}, {name:'愚园路'}, {name:'共青团'}, {name:'威海路'}
] AS lm
MATCH (l:Landmark {name:lm.name})
MERGE (t)-[:包含]->(l)
WITH t,l
MERGE (g:Grade {name:l.grade})
MERGE (l)-[:情感评级 {score:l.score}]->(g)
RETURN 'Theme 6 done' AS status;