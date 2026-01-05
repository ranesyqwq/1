// theme_1_culture.cypher - 主题：文化场馆/历史街区
// 创建主题节点并关联其打卡点（只含主题、打卡点、情感评级）
MERGE (t:Theme {name:'文化场馆/历史街区'})
WITH t
UNWIND [
  {name:'豫园'}, {name:'外白渡桥'}, {name:'城隍庙'}, {name:'田子坊'}, {name:'上海城隍庙'}, {name:'静安寺'}, {name:'龙华寺'}, {name:'朱家角'}
] AS lm
MATCH (l:Landmark {name:lm.name})
MERGE (t)-[:包含]->(l)
WITH t,l
MERGE (g:Grade {name:l.grade})
MERGE (l)-[:情感评级 {score:l.score}]->(g)
RETURN 'Theme 1 done' AS status;