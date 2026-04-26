-- Write your SQL query here
SELECT DISTINCT customer_name, COUNT(distinct product) AS unique_products from orders group by customer_name
    order by unique_products DESC, customer_name ASC;