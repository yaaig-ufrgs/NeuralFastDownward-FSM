(define (problem blocks-14-0)
(:domain blocks)
(:objects i d b l c k m h j n e f g a )
		(:init
		(clear b)
		(clear e)
		(clear f)
		(clear k)
		(handempty)
		(on a c)
		(on b a)
		(on c g)
		(on d m)
		(on f n)
		(on g i)
		(on j l)
		(on k j)
		(on m h)
		(on n d)
		(ontable e)
		(ontable h)
		(ontable i)
		(ontable l)
		)
(:goal (and (on e l) (on l f) (on f b) (on b j) (on j i) (on i n) (on n c)
            (on c k) (on k g) (on g d) (on d m) (on m a) (on a h)))
)
