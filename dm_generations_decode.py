import argparse
from collections import deque
import concurrent.futures
from dataclasses import dataclass, field, fields
from functools import partial, partialmethod
from PIL import Image
from math import ceil
import numpy as np
import os
import pathlib
from pathvalidate import sanitize_filename
import sqlite3
from tqdm import tqdm
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)


@dataclass
class ImageGroup:
    db: sqlite3.Cursor
    prompt_id: int
    prompt_text: Optional[str]
    sc_prompt_id: int
    sc_prompt_text: Optional[str]
    top_k: int
    top_p: float
    temperature: float
    cond_scale: float
    ct: int

    @property
    def embeddings(self) -> List[np.ndarray]:
        if not hasattr(self, "_embeddings"):
            self._embeddings =[
                    np.frombuffer(row["quantvec"], dtype="<u2")
                    for row in self.db.execute(
                        """
                        SELECT quantvec FROM images
                        WHERE 
                            prompt_id = :prompt_id 
                            AND sc_prompt_id = :sc_prompt_id
                            AND top_k = :top_k
                            AND top_p = :top_p
                            AND temperature = :temperature
                            AND cond_scale = :cond_scale
                        ORDER BY clip_score_input DESC
                        LIMIT :ct
                        """,
                        {f.name: getattr(self, f.name) for f in fields(self) if f.name != "db"}
                    ).fetchall()
            ]
        return self._embeddings

    @classmethod
    def fetch_meta(
        cls,
        db: sqlite3.Cursor,
        n_best: int = None,
        prompt_ids: Collection[int] = [],
        sc_prompt_ids: Collection[int] = [],
        prompt_texts: Collection[str] = [],
        sc_prompt_texts: Collection[str] = [],
    ) -> Iterable["ImageGroup"]:
        params = []
        def parameterize_list(values):
            params.extend(values)
            placeholders = ",".join("?" * len(values))
            return f"({placeholders})"

        prompt_filter = (
            "" if len(prompt_ids) == 0 and len(prompt_texts) == 0
            else f"AND (prompts.id IN {parameterize_list(prompt_ids)} OR prompts.desc IN {parameterize_list(prompt_texts)})"
        )
        sc_prompt_filter = (
            "" if len(sc_prompt_ids) == 0 and len(sc_prompt_texts) == 0
            else f"AND (sc_prompts.id IN {parameterize_list(sc_prompt_ids)} OR sc_prompts.desc IN {parameterize_list(sc_prompt_texts)})"
        )

        query = f"""
            SELECT
                prompts.id AS prompt_id,
                prompts.desc AS prompt_text,
                sc_prompts.id AS sc_prompt_id,
                sc_prompts.desc AS sc_prompt_text,
                images.top_k,
                images.top_p,
                images.temperature,
                images.cond_scale,
                COUNT(*) AS ct
            FROM images, prompts, sc_prompts
            WHERE
                images.prompt_id = prompts.id
                AND images.sc_prompt_id = sc_prompts.id
                {prompt_filter} {sc_prompt_filter}
            GROUP BY
                prompts.id,
                sc_prompts.id,
                images.top_k,
                images.top_p,
                images.temperature,
                images.cond_scale
        """

        for row in db.execute(query, params).fetchall():
            kw = {
                **row,
                "ct": row["ct"] if n_best is None else min(row["ct"], n_best),
            }
            yield cls(db, **kw)


def make_prompt_dir(out_dir: str, group: ImageGroup) -> pathlib.Path:
    prompt_name = sanitize_filename(f"{group.prompt_id:04d}_{group.prompt_text}")
    prompt_path = pathlib.Path(out_dir, prompt_name)
    prompt_path.mkdir(exist_ok=True, parents=True)
    return prompt_path


def save_image_in_group(
    out_dir: str, overwrite: bool, image: np.ndarray, rank: int, group: ImageGroup
) -> None:
    if group.ct == 0:
        return

    prompt_path = make_prompt_dir(out_dir, group)
    image_name = sanitize_filename(
        f"{group.sc_prompt_id:04d}_k={group.top_k},p={group.top_p},t={group.temperature},c={group.cond_scale}_{rank:02d}_{group.sc_prompt_text}"
    )[:250]
    image_path = (prompt_path / image_name)
    image_path = image_path.with_suffix(f"{image_path.suffix}.png")
    if image_path.exists() and not overwrite:
        return

    Image.fromarray(image).save(image_path)


def batch_to_grid(cols: int, rows: int, batch: np.ndarray) -> np.ndarray:
    h, w, c = batch.shape[-3:]
    return (
        batch.reshape((rows, cols, h, w, c))
        .transpose(0, 2, 1, 3, 4)  # (rows, h, cols, w, c)
        .reshape((rows * h, cols * w, c))
    )


def save_group_gallery(
    cols: int,
    rows: int,
    out_dir: str,
    overwrite: bool,
    group: ImageGroup,
    batches: Sequence[np.ndarray],
) -> None:
    if group.ct == 0:
        return

    prompt_path = make_prompt_dir(out_dir, group)
    image_name = sanitize_filename(
        f"{group.sc_prompt_id:04d}_k={group.top_k},p={group.top_p},t={group.temperature},c={group.cond_scale}_{group.sc_prompt_text}"
    )[:250]
    image_path = (prompt_path / image_name)
    image_path = image_path.with_suffix(f"{image_path.suffix}.png")
    if image_path.exists() and not overwrite:
        return

    # create padded (rows * cols, h, w, c) array so we don't have to resize
    for batch in batches:
        if len(batch) > 0:
            h, w, c = batch.shape[-3:]
            break
    images = np.zeros((rows * cols, h, w, c), dtype=np.uint8)
    np.concatenate(batches, out=images[: group.ct])

    Image.fromarray(batch_to_grid(cols, rows, images)).save(image_path)


ImageGroupTask = Callable[[np.ndarray], Any]
ImageGroupTaskFactory = Callable[[ImageGroup], ImageGroupTask]


@dataclass
class SaveGalleryTaskBase:
    group: ImageGroup
    decoded_batches: List[np.ndarray] = field(default_factory=list)

    def __call__(
        self, chunk: np.ndarray, executor: concurrent.futures.Executor, **args
    ) -> None:
        self.decoded_batches.append(chunk)
        if sum(batch.shape[0] for batch in self.decoded_batches) == self.group.ct:
            executor.submit(
                save_group_gallery,
                **args,
                group=self.group,
                batches=self.decoded_batches,
            )


@dataclass
class SaveImagesTaskBase:
    group: ImageGroup
    last_rank: int = 0

    def __call__(
        self, chunk: np.ndarray, executor: concurrent.futures.Executor, **args
    ) -> None:
        for img in chunk:
            rank, self.last_rank = self.last_rank, self.last_rank + 1
            executor.submit(
                save_image_in_group, **args, group=self.group, rank=rank, image=img
            )


Task = TypeVar("Task")
TaskBatch = TypeVar("TaskBatch")
TaskBatchIndex = List[Tuple[Task, int]]

class Producer:
    def __init__(self, getter):
        self.getter = getter
        self.wrapped = None
    def __call__(self, count):
        if self.wrapped is None:
            self.wrapped = self.getter()
        cur, self.wrapped = self.wrapped[:count], self.wrapped[count:]
        return cur

def batch_iter(
    queue: Iterator[Tuple[Task, TaskBatch]], batch_size: int
) -> Iterable[Tuple[TaskBatch, TaskBatchIndex]]:
    chunk = []
    try:
        while True:
            batch, index, space = [], [], batch_size
            while space > 0:
                if len(chunk) == 0:
                    task, producer = next(queue)
                chunk = producer(space)
                batch.extend(chunk)
                index.append((task, len(chunk)))
                space -= len(chunk)
            yield batch, index
    except StopIteration:
        if len(batch) > 0:
            yield batch, index


if __name__ == "__main__":
    parsers = {}

    parser = parsers["common"] = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db", type=str, default="dm_generations.db")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu-only", action="store_true", default=False)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--mode", type=str, choices=["images", "gallery"], default="images"
    )

    parser = parsers["task"] = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out-dir", type=str)
    parser.add_argument("--overwrite", action="store_true", default=False)

    parser = parsers["gallery"] = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gallery-cols", type=int, dest="cols", default=8)
    parser.add_argument("--gallery-rows", type=int, dest="rows", default=4)

    parser = parsers["filter"] = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prompt-ids", type=int, nargs="+", default=[])
    parser.add_argument("--sc-prompt-ids", type=int, nargs="+", default=[])
    parser.add_argument("--prompt-texts", type=str, nargs="+", default=[])
    parser.add_argument("--sc-prompt-texts", type=str, nargs="+", default=[])
    parser.add_argument("--n-best", type=int)

    # combined parser, for help
    parser = argparse.ArgumentParser(parents=[*parsers.values()])
    parser.parse_args()

    args = {cat: parsers[cat].parse_known_args()[0] for cat in parsers}

    # when using multiprocessing for host tasks, we do not want JAX loaded on workers
    # thus we keep all JAX code here

    # do not reserve 90% of GPU memory for JAX
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    if args["common"].cpu_only:
        os.environ["JAX_PLATFORMS"] = "cpu"

    import jax
    import jax.numpy as jnp
    from vqgan_jax.modeling_flax_vqgan import VQModel

    @jax.jit
    def vqgan_decode(indices: np.ndarray, vqgan_params):
        return (
            vqgan.decode_code(indices, params=vqgan_params).clip(0.0, 1.0) * 255
        ).astype(jnp.uint8)

    def process_groups(
        groups: Iterable[ImageGroup],
        batch_size: int,
        device,
        vqgan_params: Dict,
        task_factory: ImageGroupTaskFactory,
    ) -> None:
        total = 0
        queue = deque()
        
        getter = lambda g: g.embeddings
        for group in groups:
            task = task_factory(group)
            queue.append((task, Producer(partial(getter, group))))
            total += group.ct

        for batch, index in tqdm(
            batch_iter(iter(queue), batch_size), total=ceil(total / batch_size)
        ):
            codes = jax.device_put(batch, device)
            images = jax.device_get(vqgan_decode(codes, vqgan_params))
            i = 0
            for task, chunk_len in index:
                task(images[i : i + chunk_len])
                i += chunk_len

    VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
    VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"
    device = jax.devices()[0]

    vqgan, vqgan_params = VQModel.from_pretrained(
        VQGAN_REPO, revision=VQGAN_COMMIT_ID, dtype=jnp.float32, _do_init=False
    )
    del vqgan_params["encoder"]
    vqgan_params = jax.device_put(vqgan_params, device)

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=args["common"].num_workers
    )

    if args["task"].out_dir is None:
        args["task"].out_dir = args["common"].mode

    task_user_args = vars(args["task"])
    if args["common"].mode == "images":
        base_task = SaveImagesTaskBase
    elif args["common"].mode == "gallery":
        base_task = SaveGalleryTaskBase
        task_user_args.update(vars(args["gallery"]))
    else:
        raise ValueError(f"Unknown mode: {args['common'].mode}")

    class task_factory(base_task):
        __call__ = partialmethod(base_task.__call__, **task_user_args, executor=executor)

    con = sqlite3.connect(args["common"].db, check_same_thread=False)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    groups = ImageGroup.fetch_meta(cur, **vars(args["filter"]))
    process_groups(
        groups, args["common"].batch_size, device, vqgan_params, task_factory
    )
