(define (problem blocks-14-0)
(:domain blocks)
(:objects i d b l c k m h j n e f g a )
		(:init
		(clear a)
		(clear c)
		(clear i)
		(clear l)
		(holding e)
		(on b j)
		(on c m)
		(on f n)
		(on g h)
		(on h k)
		(on i b)
		(on l f)
		(on m g)
		(on n d)
		(ontable a)
		(ontable d)
		(ontable j)
		(ontable k)
		)
(:goal (and (on e l) (on l f) (on f b) (on b j) (on j i) (on i n) (on n c)
            (on c k) (on k g) (on g d) (on d m) (on m a) (on a h)))
)
