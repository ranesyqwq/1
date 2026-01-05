// theme_2_shopping.cypher - 主题：购物/商业街区
MERGE (t:Theme {name:'购物/商业街区'})
WITH t
UNWIND [
  {name:'南京路'}, {name:'淮海中路'}, {name:'淮海路'}, {name:'陕西南路'}, {name:'淮海路'}, {name:'淮海中路'}, {name:'徐家汇'}, {name:'南京西路'}, {name:'北京路'}
] AS lm
MATCH (l:Landmark {name:lm.name})
MERGE (t)-[:包含]->(l)
WITH t,l
MERGE (g:Grade {name:l.grade})
MERGE (l)-[:情感评级 {score:l.score}]->(g)
RETURN 'Theme 2 done' AS status;