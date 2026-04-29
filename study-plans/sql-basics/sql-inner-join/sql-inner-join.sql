-- Write your SQL query here
SELECT e.name, e.salary, d.dept_name
from employees e
inner join departments d on e.dept_id = d.id
order by e.name ASC;